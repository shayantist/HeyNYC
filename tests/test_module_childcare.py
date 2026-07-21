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

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.childcare import tools as childcare
from heynyc.modules.childcare.tools import (
    _address,
    _age_range,
    _capacity_phrase,
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


def test_capacity_phrase_is_max_not_available():
    phrase = _capacity_phrase(_record(capacity="70"))
    assert "70" in phrase
    assert "max" in phrase.lower()          # framed as maximum licensed capacity
    assert _capacity_phrase(_record(capacity="")) == ""
    assert _capacity_phrase(_record(capacity="NULL")) == ""


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
            return httpx.Response(200, json=records)
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_nearest_child_care_ranks_grounds_and_links():
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
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn", "k": 5}, ctx)
    await client.aclose()

    site_lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(site_lines) == 2                        # no-coords row dropped
    assert "Close Care" in site_lines[0]               # nearest first
    assert "Far Care" in site_lines[1]
    assert "(718) 555-0002" in out                     # phone surfaced
    assert "0 YEARS - 2 YEARS" in out                  # age range surfaced
    assert "40" in out                                 # capacity surfaced
    assert "www.google.com/maps/dir/?api=1&destination=40.69010,-73.96010" in out  # directions link
    assert "{cite:S1}" in out                          # grounded, cited
    mapping = citations.mapping()
    assert mapping["S1"]["kind"] == "DATA"
    # citation is a row-addressed permalink into the NYC Open Data dataset
    assert "data.cityofnewyork.us" in mapping["S1"]["url"]
    assert "gy3q-4tzp" in mapping["S1"]["url"]
    assert "row-close" in mapping["S1"]["url"]
    assert mapping["S1"]["provenance"]["record_id"] == "row-close"
    assert mapping["S1"]["valid_as_of"]


async def test_nearest_child_care_capacity_is_not_open_seats():
    # Honesty: capacity is the MAX licensed number, never presented as open/available seats.
    records = [_record(latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    low = out.lower()
    assert "call" in low                                # routes to calling to confirm openings/hours
    assert "available" not in low or "open spot" in low  # never claims seats are available outright


async def test_nearest_child_care_does_not_fake_missing_source_date():
    records = [_record(latitude="40.6901", longitude="-73.9601", **{":updated_at": None})]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()

    assert citations.mapping()["S1"]["valid_as_of"] == ""
    assert "Source date unavailable" in out


async def test_nearest_child_care_omits_no_data_age():
    records = [_record(age_range="NO DATA", latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert "NO DATA" not in out                          # sentinel never leaks to the user


async def test_nearest_child_care_gives_official_fallback_when_phone_is_missing():
    records = [_record(phone="", latitude="40.6901", longitude="-73.9601")]
    client = _routed_client(records)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert "nyc.gov/site/doh/services/child-care.page" in out
    assert "{cite:" in out.split("Contact:", 1)[1].splitlines()[0]
    assert any("child-care.page" in citation["url"] for citation in citations.mapping().values())


async def test_nearest_child_care_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr(childcare, "geocode", fail)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert not any(l.startswith("- ") for l in out.splitlines())   # no fabricated program list
    low = out.lower()
    assert "couldn't" in low or "could not" in low
    assert "nyc" in low


async def test_nearest_child_care_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr(childcare, "geocode", ambiguous)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert "which borough" in out.lower()
    assert not any(l.startswith("- ") for l in out.splitlines())


async def test_nearest_child_care_abstains_when_api_down():
    client = _routed_client([], status=503)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())   # don't invent when the source is down
    assert "child care" in out.lower() or "childcare" in out.lower()


async def test_nearest_child_care_abstains_when_no_programs():
    client = _routed_client([])                          # source reachable but returns nothing
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Clinton Hill Brooklyn"}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())


async def test_nearest_child_care_asks_when_location_missing():
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": ""}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())
    assert "where" in out.lower()


# --- the shipped module stays valid ---------------------------------------

def test_childcare_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "childcare"), None)
    assert module is not None
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "nearest_child_care" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "childcare"]
    assert cases, "childcare should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
