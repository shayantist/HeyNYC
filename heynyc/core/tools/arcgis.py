"""Generic ArcGIS Feature Service client.

Many NYC finders (FoodHelp, clinics, IDNYC sites, immigrant-services locations) are backed by a
public, tokenless ArcGIS Feature Service rather than Socrata. This adapter is the reusable seam
the coverage-map spec anticipated: it GETs a layer's `/query?f=geojson` endpoint and returns the
feature records as flat dicts, the same injectable shape as `datasets.query_dataset` (pass an
httpx client so tests stay fully offline).

It stays deliberately generic (not pantry-specific): each returned record is the feature's
`properties` merged with `lat`/`lon` pulled from the GeoJSON geometry (WGS84 [lon, lat]).

Ref: https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class ArcGISQueryPage:
    records: list[dict]
    exceeded_transfer_limit: bool | None
    next_offset: int | None
    pagination_token: str | None

    @property
    def complete(self) -> bool:
        return self.exceeded_transfer_limit is not True


def _feature_to_record(feature: dict) -> dict:
    """A GeoJSON feature → a flat record: its `properties`, plus `lat`/`lon` from the geometry.

    GeoJSON is always WGS84 with coordinates ordered [lon, lat]; we surface them under stable
    `lat`/`lon` keys (the authoritative coordinate, overriding any same-named property) so callers
    never have to re-parse geometry. Non-point / missing geometry leaves the record's own fields
    untouched, and the caller decides whether to drop it."""
    record = dict(feature.get("properties") or {})
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    if geometry.get("type") == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        record["lon"], record["lat"] = coords[0], coords[1]
    return record


def feature_query_url(url: str, id_value, *, id_field: str = "OBJECTID") -> str:
    """A single-feature permalink: the layer's `/query` for one row as GeoJSON.

    Row-addressed like the Socrata row permalink, a real, resolvable URL that returns exactly the
    cited feature, so a DATA citation can be re-fetched and verified. `id_field` selects the row-key
    field (defaults to the ArcGIS `OBJECTID`; pass e.g. `GlobalID` for a GUID-keyed layer). Integer-
    like values are written unquoted (`OBJECTID=5`); anything else is quoted (`GlobalID='<guid>'`).
    The whole `where` predicate is URL-encoded."""
    if str(id_value).lstrip("-").isdigit():
        predicate = f"{id_field}={id_value}"
    else:
        predicate = f"{id_field}='{id_value}'"
    return f"{url.rstrip('/')}/query?where={quote(predicate)}&outFields=*&f=geojson"


async def query_feature_service_page(
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    result_record_count: int = 2000,
    result_offset: int = 0,
    pagination_token: str | None = None,
    client: Optional[httpx.AsyncClient] = None,
) -> ArcGISQueryPage:
    """Query one ArcGIS page without discarding provider paging metadata.

    `url` is the layer URL ending in `.../FeatureServer/<n>`. Mirrors `query_dataset`'s
    injectability, pass `client` to mock the HTTP call offline.
    """
    params = {
        "where": where,
        "outFields": out_fields,
        "f": "geojson",
        "resultRecordCount": result_record_count,
    }
    if pagination_token is None:
        params["resultOffset"] = result_offset
    else:
        params["resultPaginationToken"] = pagination_token
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(f"{url.rstrip('/')}/query", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ArcGIS query response must be an object")
        features = payload.get("features")
        if not isinstance(features, list) or not all(
            isinstance(feature, dict) for feature in features
        ):
            raise ValueError("ArcGIS query features must be a list of objects")
        exceeded = payload.get("exceededTransferLimit")
        if exceeded is not None and not isinstance(exceeded, bool):
            raise ValueError("ArcGIS exceededTransferLimit must be boolean")
        token = payload.get("resultPaginationToken")
        if token is not None and not isinstance(token, str):
            raise ValueError("ArcGIS resultPaginationToken must be a string")
        return ArcGISQueryPage(
            records=[_feature_to_record(feature) for feature in features],
            exceeded_transfer_limit=exceeded,
            next_offset=(result_offset + len(features) if exceeded else None),
            pagination_token=token,
        )
    finally:
        if own_client:
            await client.aclose()


async def query_feature_service(
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    result_record_count: int = 2000,
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict]:
    """Return flat records only after exhausting the provider's typed page boundary.

    Continue with a provider-returned pagination token; otherwise use offset paging. This does not
    initiate token mode on services that require an explicit capability handshake.
    """
    records: list[dict] = []
    previous_page: list[dict] | None = None
    offset = 0
    pagination_token: str | None = None
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        while True:
            page = await query_feature_service_page(
                url,
                where=where,
                out_fields=out_fields,
                result_record_count=result_record_count,
                result_offset=offset,
                pagination_token=pagination_token,
                client=client,
            )
            if page.complete:
                return [*records, *page.records]
            if not page.records:
                raise ValueError("ArcGIS returned an empty incomplete page")
            if page.records == previous_page:
                raise ValueError("ArcGIS returned a repeated incomplete page")
            if pagination_token is not None and page.pagination_token is None:
                raise ValueError("ArcGIS incomplete token page lacks a next token")
            if page.pagination_token is not None and page.pagination_token == pagination_token:
                raise ValueError("ArcGIS returned a repeated pagination token")
            if page.pagination_token is None and page.next_offset is None:
                raise ValueError("ArcGIS incomplete page lacks a next offset")
            records.extend(page.records)
            previous_page = page.records
            if page.pagination_token is not None:
                pagination_token = page.pagination_token
            else:
                offset = page.next_offset
    finally:
        if own_client:
            await client.aclose()
