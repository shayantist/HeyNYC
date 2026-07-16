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
import pytest

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
            return httpx.Response(200, json=records)
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
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    site_lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(site_lines) == 2                        # no-coords row dropped
    assert "Close WIC Center" in site_lines[0]         # nearest first
    assert "Far WIC Center" in site_lines[1]
    assert "(718) 555-0002" in out                     # phone surfaced
    assert "http://example.org/wic" in out             # website surfaced when present
    assert "www.google.com/maps/dir/?api=1&destination=40.75100,-73.99100" in out  # directions link
    assert "{cite:S1}" in out                          # grounded, cited
    mapping = citations.mapping()
    assert mapping["S1"]["kind"] == "DATA"
    # citation is a row-addressed permalink into the NY State Socrata dataset
    assert "health.data.ny.gov" in mapping["S1"]["url"]
    assert "g4i5-r6zx" in mapping["S1"]["url"]
    assert "row-close" in mapping["S1"]["url"]
    assert mapping["S1"]["provenance"]["record_id"] == "row-close"
    assert mapping["S1"]["valid_as_of"]


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
    assert "temporary" in out.lower()                  # honesty: a rotating/temporary site is flagged


async def test_nearest_wic_site_never_invents_hours():
    # The dataset carries NO hours field - the tool must tell the user to call, never print hours.
    records = [_record(location_1={"latitude": "40.7510", "longitude": "-73.9910"})]
    client = _routed_client(records)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert "call" in out.lower()                        # routes to calling for hours/appointment


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
    assert "health.ny.gov/prevention/nutrition/wic/how_to_apply.htm" in out
    assert "{cite:" in out.split("Contact:", 1)[1].splitlines()[0]
    assert any("how_to_apply" in citation["url"] for citation in citations.mapping().values())


async def test_nearest_wic_site_does_not_fake_missing_source_date():
    records = [_record(location_1={"latitude": "40.7510", "longitude": "-73.9910"},
                       **{":updated_at": None})]
    citations = CitationRegistry()
    client = _routed_client(records)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()

    assert citations.mapping()["S1"]["valid_as_of"] == ""
    assert "Source date unavailable" in out


async def test_nearest_wic_site_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr(wic, "geocode", fail)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert not any(l.startswith("- ") for l in out.splitlines())                              # no fabricated site list
    low = out.lower()
    assert "couldn't" in low or "could not" in low
    assert "nyc" in low


async def test_nearest_wic_site_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr(wic, "geocode", ambiguous)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert "which borough" in out.lower()
    assert not any(l.startswith("- ") for l in out.splitlines())


async def test_nearest_wic_site_abstains_when_api_down():
    client = _routed_client([], wic_status=503)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())                              # don't invent a site when the source is down
    assert "wic" in out.lower()


async def test_nearest_wic_site_abstains_when_no_sites():
    client = _routed_client([])                          # source reachable but returns nothing
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square"}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())


async def test_nearest_wic_site_asks_when_location_missing():
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": ""}, ctx)
    await client.aclose()
    assert not any(l.startswith("- ") for l in out.splitlines())
    assert "where" in out.lower()


# --- the shipped module stays valid ---------------------------------------

def test_wic_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "wic"), None)
    assert module is not None
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "nearest_wic_site" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "wic"]
    assert cases, "wic should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
