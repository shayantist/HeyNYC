"""Offline tests for the generic ArcGIS Feature Service adapter.

Mirrors the injectability of datasets.query_dataset: an httpx client is injected so no
live ArcGIS call is made. This adapter is the reusable seam the coverage-map spec anticipated
(clinics / IDNYC / immigrant-services finders reuse the same ArcGIS pattern), so it stays
generic — not pantry-specific.
"""
from __future__ import annotations

import httpx
import pytest

from heynyc.core.tools.arcgis import feature_query_url, query_feature_service


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _point_feature(lon: float, lat: float, **props) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


async def test_query_feature_service_hits_geojson_query_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_fc(_point_feature(-73.99, 40.75, program="A")))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await query_feature_service(
        "https://services6.arcgis.com/x/arcgis/rest/services/Y/FeatureServer/0",
        where="status='Open'",
        client=client,
    )
    await client.aclose()

    assert seen["path"].endswith("/FeatureServer/0/query")
    assert seen["params"]["where"] == "status='Open'"
    assert seen["params"]["outFields"] == "*"
    assert seen["params"]["f"] == "geojson"
    assert seen["params"]["resultRecordCount"] == "2000"
    assert len(out) == 1 and out[0]["program"] == "A"


async def test_query_feature_service_merges_geometry_lat_lon():
    def handler(request: httpx.Request) -> httpx.Response:
        # GeoJSON coordinates are [lon, lat] in WGS84 — the adapter must surface them as lat/lon.
        return httpx.Response(200, json=_fc(_point_feature(-73.9857, 40.7484, program="Midtown Pantry")))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    [rec] = await query_feature_service("https://arcgis.example/FeatureServer/0", client=client)
    await client.aclose()

    assert rec["program"] == "Midtown Pantry"
    assert rec["lat"] == 40.7484
    assert rec["lon"] == -73.9857


async def test_query_feature_service_empty_and_no_geometry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fc(
            {"type": "Feature", "geometry": None, "properties": {"program": "No Geom"}},
        ))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await query_feature_service("https://arcgis.example/FeatureServer/0", client=client)
    await client.aclose()
    # The record is still returned (generic adapter doesn't drop it); it just lacks lat/lon.
    assert out == [{"program": "No Geom"}]

    empty = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": []})))
    assert await query_feature_service("https://arcgis.example/FeatureServer/0", client=empty) == []
    await empty.aclose()


def test_feature_query_url_is_single_object_permalink():
    url = feature_query_url("https://arcgis.example/FeatureServer/0/", "42")
    assert url == "https://arcgis.example/FeatureServer/0/query?where=OBJECTID%3D42&outFields=*&f=geojson"
