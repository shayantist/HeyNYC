"""Offline tests for the wic module.

Grounded in the NY State WIC Program Site Information dataset (Health Data NY, Socrata
`g4i5-r6zx`), but every HTTP call is mocked/injected - no live Socrata or geocoder call.
Covers: coordinate parsing from the Socrata `location` field, address assembly, ranking by
distance, the temporary-site honesty note, the directions link, a grounded row-addressed DATA
citation, and abstention when geocoding fails, the location is ambiguous, the API is down, or
no NYC sites come back.
"""
from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.wic import tools as wic
from heynyc.modules.wic.tools import (
    _address,
    _parse_location,
    _to_site,
    _valid_as_of,
    _website,
    directions_link,
    get_tools,
)

# --- pure helpers ----------------------------------------------------------

def _record(**over) -> dict:
    base = {
        "agency_name": "Test WIC Center",
        "phone_number": "(718) 555-0000",
        "street_address": "1 Test St",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "site_number": "100-01",
        "counties_boroughs_served": "New York",
        "site_type": "Permanent",
        "location_1": {"latitude": "40.7510", "longitude": "-73.9910",
                       "human_address": '{"address": "1 Test St"}'},
        ":id": "row-test1",
        ":updated_at": "2026-04-30T15:38:21.253Z",
    }
    base.update(over)
    return base


def test_parse_location_reads_lat_lon():
    assert _parse_location(_record()) == (40.7510, -73.9910)
    # blank / missing / malformed → None, never a guessed coordinate
    assert _parse_location(_record(location_1=None)) is None
    assert _parse_location(_record(location_1={})) is None
    assert _parse_location(_record(location_1={"latitude": "", "longitude": ""})) is None
    assert _parse_location({}) is None


def test_address_assembles_parts_including_street2():
    addr = _address(_record(street2="2nd Fl"))
    assert "1 Test St" in addr
    assert "2nd Fl" in addr
    assert "New York" in addr
    assert "10001" in addr


def test_website_extracted_from_url_object():
    assert _website(_record(link_to_website={"url": "http://example.org/wic"})) == "http://example.org/wic"
    assert _website(_record(link_to_website=None)) == ""
    assert _website(_record()) == ""  # key absent


def test_directions_link_is_google_maps_dir():
    link = directions_link(40.7484, -73.9857)
    assert link == "https://www.google.com/maps/dir/?api=1&destination=40.74840,-73.98570"


def test_to_site_drops_rows_without_coords():
    assert _to_site(_record()) is not None
    assert _to_site(_record(location_1=None)) is None


def test_to_site_renders_source_phone_suffix_as_extension():
    site = _to_site(_record(phone_number="(718) 839-8900+8903"))

    assert site is not None
    assert site.phone == "(718) 839-8900 ext. 8903"


def test_source_date_preserves_valid_value_and_rejects_invalid_value():
    assert _valid_as_of({":updated_at": "2026-04-30T15:38:21.253Z"}) == "2026-04-30"
    assert _valid_as_of({":updated_at": "not-a-date"}) == ""


# --- the tool handler ------------------------------------------------------

WIC_HOST = "health.data.ny.gov"
GEOSEARCH_HOST = "geosearch.planninglabs.nyc"


def _routed_client(records, wic_status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if GEOSEARCH_HOST in host:
            return httpx.Response(200, json={"features": [
                {"geometry": {"coordinates": [-73.9900, 40.7500]},
                 "properties": {"label": "Origin, Manhattan"}}]})
        if WIC_HOST in host:
            if wic_status != 200:
                return httpx.Response(wic_status, json={"error": True})
            offset = int(request.url.params.get("$offset", "0"))
            limit = int(request.url.params.get("$limit", str(wic.WIC_LIMIT)))
            return httpx.Response(200, json=records[offset:offset + limit])
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_nearest_wic_site_ranks_grounds_and_links():
    records = [
        _record(agency_name="Far WIC Center", street_address="1 Far St",
                phone_number="(718) 555-0001", site_type="Permanent",
                location_1={"latitude": "40.8000", "longitude": "-73.9600"}, **{":id": "row-far"}),
        _record(agency_name="Close WIC Center", street_address="2 Near Ave", street2="2nd Fl",
                phone_number="(718) 555-0002", site_type="Permanent",
                link_to_website={"url": "http://example.org/wic"},
                location_1={"latitude": "40.7510", "longitude": "-73.9910"}, **{":id": "row-close"}),
        _record(agency_name="No Coords WIC", location_1=None, **{":id": "row-nocoord"}),
    ]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "max_results": 5}, ctx)
    await client.aclose()

    assert [record.organization.name for record in out.records] == [
        "Close WIC Center",
        "Far WIC Center",
    ]
    assert out.records[0].origin_precision == "approximate"
    assert out.records[0].phone is not None
    assert out.records[0].phone.number == "(718) 555-0002"
    assert out.records[0].website == "http://example.org/wic"
    assert out.records[0].action_url == (
        "https://www.google.com/maps/dir/?api=1&destination=40.75100,-73.99100"
    )
    assert out.records[0].citation_id == "S2"
    mapping = citations.mapping()
    assert mapping["S2"]["kind"] == "DATA"
    # citation is a row-addressed permalink into the NY State Socrata dataset
    assert "health.data.ny.gov" in mapping["S2"]["url"]
    assert "g4i5-r6zx" in mapping["S2"]["url"]
    assert "row-close" in mapping["S2"]["url"]
    assert mapping["S2"]["provenance"]["record_id"] == "row-close"
    assert mapping["S2"]["valid_as_of"]
    assert out.application_route is not None
    assert out.application_route.url == wic.WIC_APPLY_URL
    assert mapping[out.application_route.citation_id]["url"] == wic.WIC_APPLY_URL


async def test_wic_tool_returns_typed_source_records_with_explicit_unknowns():
    client = _routed_client([_record()])
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=client,
        current_location=GeoPoint(
            40.7500,
            -73.9900,
            "Union Square, Manhattan",
            resident_query="Union Square",
            provider_id="place-1",
            provider_payload={"place_id": 1, "display_name": "private provider value"},
        ),
    )

    tool = get_tools()[0]
    result = await tool.handler({"near": "Union Square", "max_results": 1}, ctx)
    await client.aclose()

    assert tool.return_type is wic.WicResult
    assert isinstance(result, wic.WicResult)
    payload = result.model_dump(mode="json", exclude_none=False)
    assert payload["outcome"] == "success"
    assert payload["source"]["returned_count"] == 1
    assert payload["source"]["complete"] is True
    assert payload["source"]["pages_fetched"] == 1
    assert payload["source"]["page_size"] == 500
    assert "provider_payload" not in payload["origin"]
    assert payload["records"][0]["organization"]["name"] == "Test WIC Center"
    assert payload["records"][0]["hours"] is None
    assert payload["records"][0]["appointment_required"] is None
    assert payload["records"][0]["service"]["eligibility_description"] is None
    assert payload["records"][0]["citation_id"].startswith("S")
    assert payload["records"][0]["action_url"].startswith("https://www.google.com/maps/dir/")


async def test_wic_tool_reuses_current_location_without_geocoding(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("stored origin reached geocoder")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", should_not_geocode)
    client = _routed_client([_record()])
    current = GeoPoint(
        40.7500,
        -73.9900,
        "Union Square, Manhattan",
        resident_query="Union Square",
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=client,
        current_location=current,
    )

    result = await get_tools()[0].handler({"near": "Union Square", "max_results": 1}, ctx)
    await client.aclose()

    assert result.outcome == "success"
    assert ctx.current_location is current
    assert result.origin is not None
    assert result.origin.resident_query == "Union Square"


async def test_wic_tool_types_a_malformed_provider_response():
    def handler(request: httpx.Request) -> httpx.Response:
        if GEOSEARCH_HOST in request.url.host:
            return httpx.Response(200, json={"features": [
                {
                    "geometry": {"coordinates": [-73.9900, 40.7500]},
                    "properties": {"label": "Origin, Manhattan"},
                }
            ]})
        return httpx.Response(200, json={"error": "not a row list"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    result = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()

    assert result.outcome == "source_unavailable"
    assert result.source.status == "unavailable"
    assert result.source.fetched_at is not None
    assert result.source.error == "invalid_response"
    assert result.records == []


async def test_wic_query_creates_client_when_runtime_does_not_inject_one(monkeypatch):
    client = _routed_client([])
    monkeypatch.setattr(wic.httpx, "AsyncClient", lambda **_kwargs: client)

    result = await wic._query_wic(None, where=wic.WHERE_NYC)

    assert result.rows == []
    assert result.complete is True
    assert client.is_closed


async def test_wic_query_reads_all_stably_ordered_pages():
    records = [
        _record(agency_name=f"WIC {row_id}", **{":id": row_id})
        for row_id in ("row-1", "row-2", "row-3")
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params.get("$offset", "0"))
        limit = int(request.url.params["$limit"])
        return httpx.Response(200, json=records[offset:offset + limit])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await wic._query_wic(client, where=wic.WHERE_NYC, limit=2)
    await client.aclose()

    assert [row[":id"] for row in result.rows] == ["row-1", "row-2", "row-3"]
    assert result.complete is True
    assert result.pages_fetched == 2
    assert [request.url.params.get("$offset", "0") for request in requests] == ["0", "2", "3"]
    assert all(request.url.params["$order"] == ":id" for request in requests)


async def test_wic_query_preserves_partial_page_failure_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("$offset", "0"))
        if offset:
            return httpx.Response(503, json={"error": True})
        return httpx.Response(200, json=[_record(**{":id": "row-1"})])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await wic._query_wic(client, where=wic.WHERE_NYC, limit=1)
    await client.aclose()

    assert [row[":id"] for row in result.rows] == ["row-1"]
    assert result.complete is False
    assert result.pages_fetched == 1
    assert result.error == "transport_error"


async def test_wic_tool_does_not_rank_an_incomplete_citywide_result(monkeypatch):
    async def partial(*_args, **_kwargs):
        return wic._WicQueryPage(
            rows=[_record(**{":id": "row-1"})],
            pages_fetched=1,
            complete=False,
            error="transport_error",
        )

    monkeypatch.setattr(wic, "_query_wic", partial)
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    result = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()

    assert result.outcome == "source_partial"
    assert result.source.status == "partial"
    assert result.source.complete is False
    assert result.source.next_offset == 1
    assert result.records == []


async def test_nearest_wic_site_flags_temporary_site():
    records = [
        _record(agency_name="Pop-up WIC", site_type="Temporary",
                location_1={"latitude": "40.7510", "longitude": "-73.9910"}, **{":id": "row-temp"}),
    ]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert out.records[0].service_at_location.site_type == "Temporary"


async def test_nearest_wic_site_never_invents_hours():
    # The dataset carries NO hours field - the tool must tell the user to call, never print hours.
    records = [_record(location_1={"latitude": "40.7510", "longitude": "-73.9910"})]
    client = _routed_client(records)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert out.records[0].hours is None
    assert out.records[0].appointment_required is None
    assert out.records[0].phone is not None


async def test_nearest_wic_site_gives_official_fallback_when_contact_is_missing():
    records = [_record(
        phone_number="",
        link_to_website=None,
        location_1={"latitude": "40.7510", "longitude": "-73.9910"},
    )]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert out.records[0].phone is None
    assert out.records[0].website is None
    assert out.application_route is not None
    assert "how_to_apply" in out.application_route.url
    assert out.application_route.citation_id in citations.mapping()


async def test_nearest_wic_site_does_not_fake_missing_source_date():
    records = [_record(location_1={"latitude": "40.7510", "longitude": "-73.9910"},
                       **{":updated_at": None})]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()

    record_citation = citations.mapping()[out.records[0].citation_id]
    assert record_citation["valid_as_of"] == ""
    assert out.records[0].valid_as_of is None


async def test_nearest_wic_site_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fail)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert out.outcome == "location_not_found"
    assert out.records == []
    assert out.source.status == "not_called"


async def test_nearest_wic_site_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", ambiguous)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert out.outcome == "location_ambiguous"
    assert out.origin is not None and out.origin.low_confidence
    assert out.records == []


async def test_nearest_wic_site_abstains_when_api_down():
    client = _routed_client([], wic_status=503)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert out.outcome == "source_unavailable"
    assert out.source.status == "unavailable"
    assert out.source.fetched_at is not None
    assert out.source.error == "transport_error"
    assert out.records == []
    assert out.application_route is not None
    assert out.application_route.citation_id in ctx.citations.mapping()


async def test_nearest_wic_site_abstains_when_no_sites():
    client = _routed_client([])                          # source reachable but returns nothing
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert out.outcome == "no_results"
    assert out.source.status == "ok"
    assert out.source.returned_count == 0
    assert out.records == []
    assert out.application_route is not None


async def test_nearest_wic_site_asks_when_location_missing():
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": ""}, ctx)
    await client.aclose()
    assert out.outcome == "missing_origin"
    assert out.source.status == "not_called"
    assert out.records == []


# --- the shipped module stays valid ---------------------------------------

def test_wic_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "wic"), None)
    assert module is not None
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "find_wic_sites" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "wic"]
    assert cases, "wic should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
