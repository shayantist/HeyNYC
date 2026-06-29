from __future__ import annotations

import json

import httpx
import pytest

from heynyc.core.tools.datasets import Place, dataset_url, normalize, query_dataset

FIELD_MAP = {"name": "propertyname", "lat": "y", "lon": "x", "status": "status", "borough": "borough"}


def test_normalize_maps_and_skips_bad_coords():
    records = [
        {"propertyname": "Site A", "y": "40.75", "x": "-73.99", "status": "Activated", "borough": "Manhattan"},
        {"propertyname": "No Coords", "status": "Activated"},  # dropped
        {"propertyname": "Bad Coords", "y": "abc", "x": "-73.9"},  # dropped
    ]
    places = normalize(records, FIELD_MAP, source_url="u")
    assert len(places) == 1
    assert places[0] == Place(
        name="Site A", lat=40.75, lon=-73.99, address="", status="Activated",
        borough="Manhattan", source_url="u", raw=records[0],
    )


def test_dataset_url():
    assert dataset_url("h2bn-gu9k").endswith("/resource/h2bn-gu9k.json")


async def test_query_dataset_sends_soql_and_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        seen["token"] = request.headers.get("X-App-Token")
        return httpx.Response(200, json=[{"ok": 1}])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await query_dataset("abc-1234", where="status='Activated'", limit=5, client=client, app_token="TKN")
    await client.aclose()

    assert out == [{"ok": 1}]
    assert seen["params"]["$where"] == "status='Activated'"
    assert seen["params"]["$limit"] == "5"
    assert seen["token"] == "TKN"


async def test_query_dataset_sends_fulltext_q():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await query_dataset("abc-1234", q="food stamps", limit=5, client=client)
    await client.aclose()

    assert seen["params"]["$q"] == "food stamps"
    assert seen["params"]["$limit"] == "5"


async def test_query_dataset_no_token_header_when_blank():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "X-App-Token" not in request.headers
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await query_dataset("abc-1234", client=client, app_token="")
    await client.aclose()
