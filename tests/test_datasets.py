from __future__ import annotations

import httpx
import pytest

from heynyc.core.tools.datasets import (
    Place,
    dataset_url,
    normalize,
    query_dataset,
    query_dataset_pages,
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
    out = await query_dataset(
        "abc-1234",
        where="status='Activated'",
        select="status, count(*) as count",
        group="status",
        limit=5,
        client=client,
        app_token="TKN",
    )
    await client.aclose()

    assert out == [{"ok": 1}]
    assert seen["params"]["$where"] == "status='Activated'"
    assert seen["params"]["$select"] == "status, count(*) as count"
    assert seen["params"]["$group"] == "status"
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


async def test_query_dataset_sends_offset_for_provider_paging():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await query_dataset("abc-1234", offset=1000, client=client)
    await client.aclose()

    assert seen["params"]["$offset"] == "1000"


async def test_query_dataset_pages_exhausts_stably_ordered_results():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        requests.append(params)
        offset = int(params.get("$offset", 0))
        rows = [
            {":id": f"row-{index}", "startdate": f"2026-08-{index + 1:02d}"}
            for index in range(offset, min(offset + 2, 3))
        ]
        return httpx.Response(200, json=rows)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await query_dataset_pages(
        "abc-1234",
        order="startdate",
        page_size=2,
        client=client,
    )
    await client.aclose()

    assert [row[":id"] for row in result.records] == ["row-0", "row-1", "row-2"]
    assert result.complete is True
    assert result.pages_fetched == 2
    assert result.next_offset is None
    assert [request.get("$offset", "0") for request in requests] == ["0", "2"]
    assert all(request["$order"] == "startdate, :id" for request in requests)


async def test_query_dataset_pages_preserves_rows_when_a_later_page_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("$offset") == "2":
            raise httpx.ReadTimeout("later page failed")
        return httpx.Response(200, json=[{":id": "row-1"}, {":id": "row-2"}])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await query_dataset_pages("abc-1234", page_size=2, client=client)
    await client.aclose()

    assert [row[":id"] for row in result.records] == ["row-1", "row-2"]
    assert result.complete is False
    assert result.pages_fetched == 1
    assert result.next_offset == 2
    assert result.error == "transport_error"


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


async def test_query_dataset_can_omit_row_only_defaults_for_an_aggregate():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await query_dataset(
        "abc-1234",
        select="status, count(*) as count",
        group="status",
        limit=None,
        exclude_system_fields=None,
        client=client,
    )
    await client.aclose()

    assert "$limit" not in seen["params"]
    assert "$$exclude_system_fields" not in seen["params"]
