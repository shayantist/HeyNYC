"""NYC Open Data (Socrata SODA) client + normalization.

Authoritative structured data for locations/events. The `field_map` on a
module's DatasetBinding maps each dataset's real columns to a common Place
shape so geo tools stay dataset-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import httpx

from ..config import SOCRATA_APP_TOKEN, SOCRATA_BASE


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    address: str = ""
    status: str = ""
    borough: str = ""
    phone: str = ""
    source_url: str = ""
    record_id: str = ""   # per-row primary key (Socrata `:id`, or the arcgis `record_id_field`)
    updated_at: str = ""  # Socrata `:updated_at`, the row's "as of" / change signal (blank for arcgis)
    raw: dict = field(default_factory=dict)


def dataset_url(dataset_id: str) -> str:
    return f"{SOCRATA_BASE}/{dataset_id}.json"


def row_url(dataset_id: str, row_id: str) -> str:
    """Single-row permalink: /resource/{4x4}/{:id}.json (Socrata row identifier)."""
    return f"{SOCRATA_BASE}/{dataset_id}/{row_id}.json"


def _get(record: dict, field_map: dict, key: str, default: str = "") -> str:
    column = field_map.get(key)
    if not column:
        return default
    value = record.get(column, default)
    return default if value is None else value


def normalize(
    records: list[dict],
    field_map: dict,
    source_url: str = "",
    record_id_field: Optional[str] = None,
) -> list[Place]:
    """Map raw records (Socrata or arcgis) to Places. Records without usable coords are dropped.

    `record_id_field` names the stable row-key column for arcgis sources (e.g. "NYCEM_ID");
    when omitted we fall back to the Socrata `:id` system field.
    """
    places: list[Place] = []
    for record in records:
        try:
            lat = float(_get(record, field_map, "lat"))
            lon = float(_get(record, field_map, "lon"))
        except (TypeError, ValueError):
            continue
        if record_id_field:
            record_id = str(record.get(record_id_field, ""))
        else:
            record_id = str(record.get(":id", ""))
        places.append(
            Place(
                name=str(_get(record, field_map, "name")),
                lat=lat,
                lon=lon,
                address=str(_get(record, field_map, "address")),
                status=str(_get(record, field_map, "status")),
                borough=str(_get(record, field_map, "borough")),
                phone=str(_get(record, field_map, "phone")),
                source_url=source_url,
                record_id=record_id,
                updated_at=str(record.get(":updated_at", "")),
                raw=record,
            )
        )
    return places


async def query_dataset(
    dataset_id: str,
    *,
    where: Optional[str] = None,
    select: Optional[str] = None,
    order: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 1000,
    client: Optional[httpx.AsyncClient] = None,
    app_token: Optional[str] = None,
) -> list[dict]:
    """Run a SoQL query against a Socrata dataset and return raw records."""
    # `$$exclude_system_fields=false` returns the `:id` / `:updated_at` system fields so each
    # row is addressable (row permalink) and carries its own "as of" date.
    params: dict = {"$limit": limit, "$$exclude_system_fields": "false"}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order
    if q:
        params["$q"] = q

    headers: dict = {}
    token = app_token if app_token is not None else SOCRATA_APP_TOKEN
    if token:
        headers["X-App-Token"] = token

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(dataset_url(dataset_id), params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    finally:
        if own_client:
            await client.aclose()
