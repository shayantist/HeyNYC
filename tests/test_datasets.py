from __future__ import annotations

import httpx
import pytest

from heynyc.core.tools.datasets import (
    Place,
    dataset_url,
    normalize,
    query_dataset,
    row_url,
)

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


def test_row_url_is_single_row_permalink():
    assert row_url("h2bn-gu9k", "row-abc.123").endswith("/resource/h2bn-gu9k/row-abc.123.json")


def test_normalize_captures_system_fields():
    records = [{":id": "row-1", ":updated_at": "2026-06-20T00:00:00.000",
                "propertyname": "Marconi Park", "y": "40.7", "x": "-73.8",
                "status": "Activated", "borough": "Queens"}]
    [place] = normalize(records, FIELD_MAP, source_url="https://x")
    assert place.record_id == "row-1" and place.updated_at == "2026-06-20T00:00:00.000"
    assert place.raw[":id"] == "row-1"


@pytest.mark.parametrize("updated_at", [None, "not-a-date"])
def test_normalize_omits_invalid_system_timestamp(updated_at):
    records = [{
        ":updated_at": updated_at,
        "propertyname": "Marconi Park",
        "y": "40.7",
        "x": "-73.8",
    }]
    [place] = normalize(records, FIELD_MAP)
    assert place.updated_at == ""


def test_normalize_maps_socrata_url_to_website():
    records = [{
        "propertyname": "53rd Street Library",
        "y": "40.76082",
        "x": "-73.97737",
        "website": {"url": "https://www.nypl.org/locations/53rd-street"},
        "comments": "Monday-Friday 8:30am to 5:00pm",
    }]
    [place] = normalize(records, {**FIELD_MAP, "website": "website", "hours": "comments"})
    assert place.website == "https://www.nypl.org/locations/53rd-street"
    assert place.hours == "Monday-Friday 8:30am to 5:00pm"


async def test_query_dataset_requests_system_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await query_dataset("abc-1234", client=client)
    await client.aclose()
    assert seen["params"]["$$exclude_system_fields"] == "false"
