"""Offline tests for the childcare module.

Grounded in the DOHMH "Active NYC Health Code Regulated Child Care Programs" dataset (NYC Open
Data, Socrata `gy3q-4tzp`), but every HTTP call is mocked/injected - no live Socrata or geocoder
call. Covers: coordinate parsing from the flat lat/lon fields, address assembly, the facility /
program / age-range / capacity labels, ranking by distance, the "max licensed capacity, not open
seats" honesty note, the directions link, a grounded row-addressed DATA citation, and abstention
when geocoding fails, the location is ambiguous, the API is down, or no programs come back.
"""
from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.childcare import tools as childcare
from heynyc.modules.childcare.tools import (
    _address,
    _age_range,
    _capacity,
    _facility_label,
    _parse_coords,
    _program_label,
    _to_site,
    _valid_as_of,
    directions_link,
    get_tools,
)

# --- pure helpers ----------------------------------------------------------

def _record(**over) -> dict:
    base = {
        "dcid": "DC1000",
        "permit_number": "2165",
        "program_name": "Billy Martin Child Development Center",
        "facility_type": "GCC",
        "program_type": "PRESCHOOL",
        "address": "333 CLASSON AVENUE",
        "borough": "BROOKLYN",
        "zipcode": "11205",
        "phone": "(718) 857-5630",
        "age_range": "2 YEARS - 5 YEARS",
        "capacity": "70",
        "administer_medication": "No",
        "latitude": "40.689761",
        "longitude": "-73.960169",
        ":id": "row-test1",
        ":updated_at": "2026-05-14T14:50:58.844Z",
    }
    base.update(over)
    return base


def test_parse_coords_reads_lat_lon():
    assert _parse_coords(_record()) == (40.689761, -73.960169)
    # blank / missing / malformed -> None, never a guessed coordinate
    assert _parse_coords(_record(latitude=None, longitude=None)) is None
    assert _parse_coords(_record(latitude="", longitude="")) is None
    assert _parse_coords({}) is None


def test_address_assembles_parts():
    addr = _address(_record())
    assert "333 CLASSON AVENUE" in addr
    assert "BROOKLYN" in addr
    assert "11205" in addr


def test_facility_and_program_labels_come_from_source_codes():
    assert _facility_label(_record(facility_type="GCC")) == "group child care"
    assert _facility_label(_record(facility_type="SBCC")) == "school-based child care"
    assert _facility_label(_record(facility_type="ZZZ")) == ""  # unknown code -> no invented label
    assert _program_label(_record(program_type="PRESCHOOL")) == "preschool"
    assert _program_label(_record(program_type="INFANT TODDLER")) == "infant/toddler"


def test_age_range_drops_no_data():
    assert _age_range(_record(age_range="0 YEARS - 2 YEARS")) == "0 YEARS - 2 YEARS"
    assert _age_range(_record(age_range="NO DATA")) == ""     # source sentinel -> blank, never shown
    assert _age_range(_record(age_range=None)) == ""


def test_capacity_is_typed_and_missing_values_remain_unknown():
    assert _capacity(_record(capacity="70")) == 70
    assert _capacity(_record(capacity="")) is None
    assert _capacity(_record(capacity="NULL")) is None
    assert _capacity(_record(capacity="not a number")) is None


def test_directions_link_is_google_maps_dir():
    link = directions_link(40.7484, -73.9857)
    assert link == "https://www.google.com/maps/dir/?api=1&destination=40.74840,-73.98570"


def test_to_site_drops_rows_without_coords():
    assert _to_site(_record()) is not None
    assert _to_site(_record(latitude=None)) is None


def test_source_date_preserves_valid_value_and_rejects_invalid_value():
    assert _valid_as_of({":updated_at": "2026-05-14T14:50:58.844Z"}) == "2026-05-14"
    assert _valid_as_of({":updated_at": "not-a-date"}) == ""


# --- the tool handler ------------------------------------------------------

CHILDCARE_HOST = "data.cityofnewyork.us"
GEOSEARCH_HOST = "geosearch.planninglabs.nyc"
BOROUGH_BOUNDARY_HOST = "services5.arcgis.com"


def _routed_client(records, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if GEOSEARCH_HOST in host:
            return httpx.Response(200, json={"features": [
                {"geometry": {"coordinates": [-73.9600, 40.6900]},
                 "properties": {"label": "Origin, Brooklyn"}}]})
        if BOROUGH_BOUNDARY_HOST in host:
            return httpx.Response(200, json={"count": 1})
        if CHILDCARE_HOST in host:
            if status != 200:
                return httpx.Response(status, json={"error": True})
            offset = int(request.url.params.get("$offset", "0"))
            limit = int(request.url.params.get("$limit", str(len(records))))
            return httpx.Response(200, json=records[offset:offset + limit])
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_find_child_care_connect_programs_ranks_grounds_and_links():
    records = [
        _record(program_name="Far Care", address="1 FAR ST", phone="(718) 555-0001",
                latitude="40.8000", longitude="-73.9600", **{":id": "row-far"}),
        _record(program_name="Close Care", address="2 NEAR AVE", phone="(718) 555-0002",
                facility_type="SBCC", program_type="INFANT TODDLER", age_range="0 YEARS - 2 YEARS",
                capacity="40", latitude="40.6901", longitude="-73.9601", **{":id": "row-close"}),
        _record(program_name="No Coords Care", latitude=None, longitude=None, **{":id": "row-nocoord"}),
    ]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn", "max_results": 5}, ctx)
    await client.aclose()

    assert isinstance(out, BaseModel)
    assert out.outcome == "source_partial"
    assert len(out.programs) == 2
    assert out.programs[0].organization.name == "Close Care"
    assert out.programs[1].organization.name == "Far Care"
    assert out.origin is not None and out.origin.precision == "approximate"
    assert out.programs[0].service.age_range == "0 YEARS - 2 YEARS"
    assert out.programs[0].service.licensed_capacity == 40
    assert out.programs[0].service.current_openings is None
    assert out.programs[0].service.language is None
    assert out.programs[0].service.accessibility is None
    assert out.programs[0].service.service_area is None
    assert out.programs[0].service.eligibility is None
    assert out.programs[0].service.required_document is None
    assert out.programs[0].service_at_location.schedule is None
    assert out.programs[0].phone is not None
    assert out.programs[0].phone.number == "(718) 555-0002"
    assert out.programs[0].action_url.endswith("40.69010,-73.96010")
    assert out.programs[0].citation_id == "S1"
    assert out.source.complete is False
    assert out.source.query_filter == childcare.WHERE_HAS_COORDS
    assert out.source.includes_rows_without_coordinates is False
    assert out.source.excluded_row_count == 1
    mapping = citations.mapping()
    assert mapping["S1"]["kind"] == "DATA"
    # citation is a row-addressed permalink into the NYC Open Data dataset
    assert "data.cityofnewyork.us" in mapping["S1"]["url"]
    assert "gy3q-4tzp" in mapping["S1"]["url"]
    assert "row-close" in mapping["S1"]["url"]
    assert mapping["S1"]["provenance"]["record_id"] == "row-close"
    assert mapping["S1"]["valid_as_of"]


async def test_childcare_query_pages_until_provider_is_exhausted(monkeypatch):
    rows = [_record(**{":id": f"row-{index}"}) for index in range(3)]
    offsets = []

    async def paged_query(_dataset_id, **kwargs):
        offsets.append(kwargs["offset"])
        start = kwargs["offset"]
        return rows[start:start + kwargs["limit"]]

    monkeypatch.setattr(childcare, "query_dataset", paged_query)
    page = await childcare._query_childcare(None, page_size=2)

    assert page.rows == rows
    assert page.pages_fetched == 2
    assert page.complete is True
    assert offsets == [0, 2, 3]


async def test_childcare_query_preserves_partial_rows_when_later_page_fails(monkeypatch):
    rows = [_record(**{":id": f"row-{index}"}) for index in range(2)]

    async def failing_second_page(_dataset_id, **kwargs):
        if kwargs["offset"] == 0:
            return rows
        raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr(childcare, "query_dataset", failing_second_page)
    page = await childcare._query_childcare(None, page_size=2)

    assert page.rows == rows
    assert page.pages_fetched == 1
    assert page.complete is False
    assert page.error == "transport_error"


async def test_childcare_handler_preserves_ranked_rows_from_an_incomplete_page(monkeypatch):
    async def partial_page(_client):
        return childcare._ChildCareQueryPage(
            rows=[_record()],
            pages_fetched=1,
            complete=False,
            error="transport_error",
        )

    monkeypatch.setattr(childcare, "_query_childcare", partial_page)
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()

    assert out.outcome == "source_partial"
    assert out.source.status == "partial"
    assert out.source.returned_count == 1
    assert out.source.complete is False
    assert out.source.page_size == childcare.CHILDCARE_PAGE_SIZE
    assert out.source.next_offset == 1
    assert len(out.programs) == 1
    assert out.programs[0].organization.name == "Billy Martin Child Development Center"
    assert out.primary_citation_id == out.programs[0].citation_id


async def test_childcare_query_rejects_repeated_provider_page(monkeypatch):
    rows = [_record(**{":id": f"row-{index}"}) for index in range(2)]

    async def repeated_query(_dataset_id, **_kwargs):
        return rows

    monkeypatch.setattr(childcare, "query_dataset", repeated_query)
    page = await childcare._query_childcare(None, page_size=2)

    assert page.rows == rows
    assert page.complete is False
    assert page.error == "invalid_response"


async def test_childcare_query_rejects_duplicate_ids_within_one_page(monkeypatch):
    duplicate = _record(**{":id": "row-duplicate"})

    async def duplicate_page(_dataset_id, **_kwargs):
        return [duplicate, duplicate]

    monkeypatch.setattr(childcare, "query_dataset", duplicate_page)
    with pytest.raises(ValueError, match="stable unique row IDs"):
        await childcare._query_childcare(None, page_size=2)


async def test_find_child_care_connect_programs_capacity_is_not_open_seats():
    # Honesty: capacity is the MAX licensed number, never presented as open/available seats.
    records = [_record(latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert out.programs[0].service.licensed_capacity == 70
    assert out.programs[0].service.current_openings is None
    assert out.scope.licensed_capacity_means_open_spots is False


async def test_find_child_care_connect_programs_does_not_fake_missing_source_date():
    records = [_record(latitude="40.6901", longitude="-73.9601", **{":updated_at": None})]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()

    assert citations.mapping()["S1"]["valid_as_of"] == ""
    assert out.programs[0].valid_as_of is None


async def test_find_child_care_connect_programs_omits_no_data_age():
    records = [_record(age_range="NO DATA", latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert out.programs[0].service.age_range is None


async def test_find_child_care_connect_programs_reuses_current_location(monkeypatch):
    async def should_not_geocode(*_args, **_kwargs):
        raise AssertionError("current location should be reused")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", should_not_geocode)
    client = _routed_client([_record(latitude="40.6901", longitude="-73.9601")])
    current = GeoPoint(
        40.69,
        -73.96,
        "Clinton Hill, Brooklyn",
        resident_query="Clinton Hill Brooklyn",
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        http=client,
        current_location=current,
    )
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()

    assert out.outcome == "success"
    assert out.origin is not None and out.origin.resident_query == "Clinton Hill Brooklyn"


async def test_find_child_care_connect_programs_types_malformed_provider_response():
    def handler(request: httpx.Request) -> httpx.Response:
        if GEOSEARCH_HOST in request.url.host:
            return httpx.Response(200, json={"features": [
                {
                    "geometry": {"coordinates": [-73.9600, 40.6900]},
                    "properties": {"label": "Origin, Brooklyn"},
                }
            ]})
        return httpx.Response(200, json={"error": "not a row list"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()

    assert out.outcome == "source_unavailable"
    assert out.source.error == "invalid_response"
    assert out.source.fetched_at is not None


async def test_find_child_care_connect_programs_gives_official_fallback_when_phone_is_missing():
    records = [_record(phone="", latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert out.programs[0].phone is None
    assert out.directory_route is not None
    route = citations.mapping()[out.directory_route.citation_id]
    assert "child-care.page" in route["url"]


async def test_find_child_care_connect_programs_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fail)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert out.outcome == "location_not_found"
    assert out.programs == []
    assert out.directory_route is not None


async def test_find_child_care_connect_programs_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", ambiguous)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert out.outcome == "location_ambiguous"
    assert out.origin is not None and out.origin.low_confidence
    assert out.programs == []


async def test_find_child_care_connect_programs_abstains_when_api_down():
    client = _routed_client([], status=503)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert out.outcome == "source_unavailable"
    assert out.source.error == "transport_error"
    assert out.source.fetched_at is not None
    assert out.programs == []


async def test_find_child_care_connect_programs_abstains_when_no_programs():
    client = _routed_client([])                          # source reachable but returns nothing
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert out.outcome == "no_results"
    assert out.source.complete is True
    assert out.programs == []


async def test_find_child_care_connect_programs_asks_when_location_missing():
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": ""}, ctx)
    await client.aclose()
    assert out.outcome == "missing_origin"
    assert out.source.status == "not_called"
    assert out.programs == []


# --- the shipped module stays valid ---------------------------------------

def test_childcare_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "childcare"), None)
    assert module is not None
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "find_child_care_connect_programs" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "childcare"]
    assert cases, "childcare should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
