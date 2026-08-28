"""Offline tests for the generic ArcGIS Feature Service adapter.

Mirrors the injectability of datasets.query_dataset: an httpx client is injected so no
live ArcGIS call is made. This adapter is the reusable seam the coverage-map spec anticipated
(clinics / IDNYC / immigrant-services finders reuse the same ArcGIS pattern), so it stays
generic, not pantry-specific.
"""
from __future__ import annotations

import httpx
import pytest

from heynyc.core.tools.arcgis import (
    feature_query_url,
    query_feature_service,
    query_feature_service_page,
    query_feature_service_result,
)


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


async def test_query_feature_service_page_preserves_provider_paging_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _fc(_point_feature(-73.99, 40.75, program="A"))
        payload.update({
            "exceededTransferLimit": True,
            "resultPaginationToken": "next-page-token",
        })
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    page = await query_feature_service_page(
        "https://arcgis.example/FeatureServer/0",
        result_record_count=25,
        result_offset=50,
        client=client,
    )
    await client.aclose()

    assert page.records[0]["program"] == "A"
    assert page.exceeded_transfer_limit is True
    assert page.complete is False
    assert page.next_offset == 51
    assert page.pagination_token == "next-page-token"


async def test_query_feature_service_exhausts_provider_pages():
    offsets = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["resultOffset"])
        offsets.append(offset)
        payload = _fc(
            _point_feature(
                -73.99 + offset,
                40.75,
                OBJECTID=offset + 1,
            )
        )
        payload["exceededTransferLimit"] = offset == 0
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    records = await query_feature_service(
        "https://arcgis.example/FeatureServer/0",
        result_record_count=2,
        client=client,
    )
    await client.aclose()

    assert offsets == [0, 1]
    assert [record["OBJECTID"] for record in records] == [1, 2]


async def test_query_feature_service_uses_provider_pagination_token_without_offset():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        requests.append(params)
        if len(requests) == 1:
            payload = _fc(_point_feature(-73.99, 40.75, OBJECTID=1))
            payload.update({
                "exceededTransferLimit": True,
                "resultPaginationToken": "next-page-token",
            })
            return httpx.Response(200, json=payload)
        assert params["resultPaginationToken"] == "next-page-token"
        assert "resultOffset" not in params
        payload = _fc(_point_feature(-73.98, 40.76, OBJECTID=2))
        payload["exceededTransferLimit"] = False
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    records = await query_feature_service(
        "https://arcgis.example/FeatureServer/0",
        result_record_count=1,
        client=client,
    )
    await client.aclose()

    assert [record["OBJECTID"] for record in records] == [1, 2]


async def test_query_feature_service_result_preserves_records_when_a_later_page_fails():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            payload = _fc(_point_feature(-73.99, 40.75, OBJECTID=1))
            payload["exceededTransferLimit"] = True
            return httpx.Response(200, json=payload)
        return httpx.Response(503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await query_feature_service_result(
        "https://arcgis.example/FeatureServer/0",
        result_record_count=1,
        client=client,
    )
    await client.aclose()

    assert [record["OBJECTID"] for record in result.records] == [1]
    assert result.complete is False
    assert result.pages_fetched == 1
    assert result.error == "transport_error"


async def test_query_feature_service_rejects_an_empty_incomplete_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "type": "FeatureCollection",
                "features": [],
                "exceededTransferLimit": True,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="incomplete page"):
        await query_feature_service(
            "https://arcgis.example/FeatureServer/0",
            result_record_count=1,
            client=client,
        )
    await client.aclose()


async def test_query_feature_service_wrapper_rejects_a_partial_result():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _fc(_point_feature(-73.99, 40.75, OBJECTID=1))
        payload["exceededTransferLimit"] = True
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="partial records"):
        await query_feature_service(
            "https://arcgis.example/FeatureServer/0",
            result_record_count=1,
            client=client,
        )
    await client.aclose()


async def test_query_feature_service_result_preserves_records_on_repeated_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _fc(_point_feature(-73.99, 40.75, OBJECTID=1))
        payload["exceededTransferLimit"] = True
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await query_feature_service_result(
        "https://arcgis.example/FeatureServer/0",
        result_record_count=1,
        client=client,
    )
    await client.aclose()

    assert [record["OBJECTID"] for record in result.records] == [1]
    assert result.complete is False
    assert result.pages_fetched == 1
    assert result.error == "invalid_response"


async def test_query_feature_service_page_rejects_missing_features():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"foo": "bar"}))
    )

    with pytest.raises(ValueError, match="features"):
        await query_feature_service_page("https://arcgis.example/FeatureServer/0", client=client)
    await client.aclose()


async def test_query_feature_service_merges_geometry_lat_lon():
    def handler(request: httpx.Request) -> httpx.Response:
        # GeoJSON coordinates are [lon, lat] in WGS84; surface them as lat/lon
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


def test_feature_query_url_supports_custom_id_field_and_types():
    # A GUID GlobalID → a quoted, URL-encoded predicate: where=GlobalID='<guid>'.
    guid = "200d88be-6abe-44ff-a13a-769d8c9ef9a1"
    url = feature_query_url("https://arcgis.example/FeatureServer/0", guid, id_field="GlobalID")
    assert f"where=GlobalID%3D%27{guid}%27" in url
    assert url.endswith("&f=geojson")

    # The default integer case stays OBJECTID and unquoted (just URL-encoded).
    default = feature_query_url("https://arcgis.example/FeatureServer/0", 5)
    assert "where=OBJECTID%3D5" in default
    assert default.endswith("&f=geojson")
