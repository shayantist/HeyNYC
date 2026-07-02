"""Offline tests for the housing module's `hpd_building_lookup` tool.

Grounded in two NYC Open Data (Socrata) HPD datasets — Housing Maintenance Code
Complaints (ygpa-z7cr) and Violations (wvxf-dwi5) — but every HTTP call is
mocked/injected: the geocoder is monkeypatched to hand back a fixed GeoPoint
(carrying the building's BBL), and the Socrata queries are served by a
MockTransport routed by dataset id. No live geocoder or Socrata call.

Covers: the grounded happy path (per-category / per-class counts, HEAT/HOT WATER
and class C call-outs, a DATA citation per dataset addressed to the filtered
query), abstention when the address has no building BBL (a bare ZIP /
neighborhood), the BBL → boro/block/lot decomposition surfaced in the citation
URL, and that the shipped module still loads with its tool + eval cases.
"""
from __future__ import annotations

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.housing import tools as housing
from heynyc.modules.housing.tools import COMPLAINTS_ID, VIOLATIONS_ID, get_tools

# 617 Courtlandt Ave, Bronx — a real building (verified live): BBL 2024110048,
# which decomposes to boro 2, block 2411, lot 48.
BBL = "2024110048"
LABEL = "617 COURTLANDT AVENUE, Bronx"


def _fixed_geocode(point):
    """An injectable stand-in for the module's `geocode` returning a fixed GeoPoint."""
    async def fn(text, **kwargs):
        return point
    return fn


def _socrata_client(complaints: list[dict], violations: list[dict]) -> httpx.AsyncClient:
    """MockTransport routing the two Socrata dataset queries by their 4x4 id in the path.

    `query_dataset` fetches full OPEN rows (no $group aggregate) and counts them client-side,
    so each canned response is just a JSON list of rows."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if COMPLAINTS_ID in path:
            return httpx.Response(200, json=complaints)
        if VIOLATIONS_ID in path:
            return httpx.Response(200, json=violations)
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# Canned OPEN rows. Complaints span a few major_category values (2 HEAT/HOT WATER);
# violations span classes A/B/C (2 class C). Dates drive the "most recent" line + as-of.
_COMPLAINTS = [
    {"major_category": "HEAT/HOT WATER", "complaint_status": "OPEN", "received_date": "2026-06-01T06:50:52.000"},
    {"major_category": "HEAT/HOT WATER", "complaint_status": "OPEN", "received_date": "2026-06-15T09:12:00.000"},
    {"major_category": "PLUMBING", "complaint_status": "OPEN", "received_date": "2026-05-20T11:00:00.000"},
    {"major_category": "PAINT/PLASTER", "complaint_status": "OPEN", "received_date": "2026-04-10T08:00:00.000"},
]
_VIOLATIONS = [
    {"class": "C", "violationstatus": "Open", "novissueddate": "2026-06-20T00:00:00.000"},
    {"class": "C", "violationstatus": "Open", "novissueddate": "2026-05-05T00:00:00.000"},
    {"class": "B", "violationstatus": "Open", "novissueddate": "2026-03-01T00:00:00.000"},
    {"class": "B", "violationstatus": "Open", "novissueddate": "2026-02-01T00:00:00.000"},
    {"class": "B", "violationstatus": "Open", "novissueddate": "2026-01-15T00:00:00.000"},
    {"class": "A", "violationstatus": "Open", "novissueddate": "2025-12-01T00:00:00.000"},
]


async def _run_lookup(monkeypatch, *, geopoint, complaints, violations):
    """Monkeypatch the module geocoder + serve Socrata via MockTransport, run the tool."""
    monkeypatch.setattr(housing, "geocode", _fixed_geocode(geopoint))
    citations = CitationRegistry()
    client = _socrata_client(complaints, violations)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"address": "617 Courtlandt Ave, Bronx"}, ctx)
    await client.aclose()
    return out, citations


# --- 1. happy path: grounded counts + call-outs + a DATA citation per dataset ---

async def test_hpd_building_lookup_grounds_counts_callouts_and_cites(monkeypatch):
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_lookup(monkeypatch, geopoint=origin,
                                       complaints=_COMPLAINTS, violations=_VIOLATIONS)

    # Building + BBL surfaced
    assert LABEL in out
    assert BBL in out

    # Totals + the specific call-outs the tool must not swallow
    assert "Open HPD complaints: 4 total" in out
    assert "2 HEAT/HOT WATER" in out
    assert "Open HPD violations: 6 total" in out
    assert "2 class C" in out

    # Per-category / per-class breakdowns are grounded, not just the totals
    assert "HEAT/HOT WATER: 2" in out
    assert "PLUMBING: 1" in out
    assert "PAINT/PLASTER: 1" in out
    assert "C: 2" in out
    assert "B: 3" in out
    assert "A: 1" in out

    # Most-recent dates come from the rows, never invented
    assert "2026-06-15" in out   # latest complaint
    assert "2026-06-20" in out   # latest violation

    # One DATA citation per dataset, inline-cited, in registration order (complaints first).
    mapping = citations.mapping()
    assert len(mapping) == 2
    assert "{cite:S1}" in out and "{cite:S2}" in out

    complaints_cite = mapping["S1"]
    violations_cite = mapping["S2"]
    assert complaints_cite["kind"] == "DATA"
    assert violations_cite["kind"] == "DATA"

    # Complaints citation is addressed to the filtered query on the full BBL.
    assert COMPLAINTS_ID in complaints_cite["url"]
    assert BBL in complaints_cite["url"]
    # Violations citation is addressed to the numeric boro/block/lot decomposition.
    assert VIOLATIONS_ID in violations_cite["url"]
    assert "2411" in violations_cite["url"]   # block
    assert "48" in violations_cite["url"]     # lot

    # Provenance is grounded in the actual rows (record id = BBL, real counts in the snapshot).
    assert complaints_cite["provenance"]["record_id"] == BBL
    assert complaints_cite["provenance"]["snapshot"]["open_complaints"] == 4
    assert complaints_cite["provenance"]["snapshot"]["heat_hot_water"] == 2
    assert violations_cite["provenance"]["snapshot"]["open_violations"] == 6
    assert violations_cite["provenance"]["snapshot"]["class_c"] == 2
    # The as-of date is the most recent row date, not fetch time.
    assert complaints_cite["valid_as_of"] == "2026-06-15"
    assert violations_cite["valid_as_of"] == "2026-06-20"


# --- 2. abstention: no building BBL (a bare ZIP / neighborhood) -------------

async def test_hpd_building_lookup_abstains_without_bbl(monkeypatch):
    # A ZIP/neighborhood geocode carries no PAD bbl → the lookup is building-level, so abstain.
    no_bbl = GeoPoint(lat=40.8195, lon=-73.9160, label="ZIP 10451 area", bbl="")
    out, citations = await _run_lookup(monkeypatch, geopoint=no_bbl,
                                       complaints=_COMPLAINTS, violations=_VIOLATIONS)

    low = out.lower()
    assert "couldn't" in low or "could not" in low
    assert "street address" in low          # asks for a specific street address
    # No fabricated record: no counts, no building line, no citation.
    assert "Open HPD" not in out
    assert "Building:" not in out
    assert "{cite:" not in out
    assert len(citations) == 0


# --- 3. BBL decomposition: boro/block/lot derived from the 10-char BBL ------

async def test_bbl_decomposes_into_boro_block_lot_in_violations_url(monkeypatch):
    # "2024110048" → boro 2, block int('02411')=2411, lot int('0048')=48. The violations
    # query keys on these numeric parts (leading zeros stripped); assert they reach the URL.
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    _out, citations = await _run_lookup(monkeypatch, geopoint=origin,
                                        complaints=_COMPLAINTS, violations=_VIOLATIONS)
    violations_url = citations.mapping()["S2"]["url"]
    assert "block" in violations_url and "2411" in violations_url
    assert "lot" in violations_url and "48" in violations_url
    assert "boroid" in violations_url        # boro digit 2 keyed as boroid


# --- 4. the shipped module stays valid -------------------------------------

def test_housing_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "housing"), None)
    assert module is not None
    assert module.category == "housing"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "hpd_building_lookup" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "housing"]
    assert cases, "housing should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
