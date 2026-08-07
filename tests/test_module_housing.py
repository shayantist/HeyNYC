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

import re
from datetime import date

import httpx

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.grounding import check_grounding
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.housing import tools as housing
from heynyc.modules.housing.tools import (
    COMPLAINTS_ID,
    LITIGATIONS_ID,
    VIOLATIONS_ID,
    _single_adult_men_intake,
    get_tools,
)

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
    {"complaint_id": "1", "major_category": "HEAT/HOT WATER", "complaint_status": "OPEN", "received_date": "2026-06-01T06:50:52.000"},
    {"complaint_id": "2", "major_category": "HEAT/HOT WATER", "complaint_status": "OPEN", "received_date": "2026-06-15T09:12:00.000"},
    {"complaint_id": "3", "major_category": "PLUMBING", "complaint_status": "OPEN", "received_date": "2026-05-20T11:00:00.000"},
    {"complaint_id": "4", "major_category": "PAINT/PLASTER", "complaint_status": "OPEN", "received_date": "2026-04-10T08:00:00.000"},
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
    assert "2 heat/hot-water" in out
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


# --- 3b. HEATING coverage: open heat complaints coded 'HEATING', not just 'HEAT/HOT WATER' ---
#
# ygpa-z7cr codes some OPEN heat complaints under a separate major_category 'HEATING'
# (minor categories HEAT RELATED / RADIATOR / HEAT-PLANT / SPACE HEATER, all genuinely heat),
# alongside 'HEAT/HOT WATER'. Verified live: citywide OPEN complaints split 2,634 HEAT/HOT WATER +
# 422 HEATING. Counting only HEAT/HOT WATER undercounts heat, so the call-out folds both in.

async def test_hpd_lookup_counts_heating_category_as_heat(monkeypatch):
    complaints = [
        {"complaint_id": "1", "major_category": "HEAT/HOT WATER", "complaint_status": "OPEN", "received_date": "2026-06-01T06:50:52.000"},
        {"complaint_id": "2", "major_category": "HEATING", "complaint_status": "OPEN", "received_date": "2026-06-10T06:50:52.000"},
        {"complaint_id": "2", "major_category": "HEATING", "complaint_status": "OPEN", "received_date": "2026-06-12T06:50:52.000"},
        {"complaint_id": "3", "major_category": "PLUMBING", "complaint_status": "OPEN", "received_date": "2026-05-20T11:00:00.000"},
    ]
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_lookup(monkeypatch, geopoint=origin, complaints=complaints, violations=[])
    # Three heat problem rows belong to two distinct complaints.
    assert "2 heat/hot-water" in out
    # both categories stay visible in the grounded per-category breakdown, not silently merged
    assert "HEAT/HOT WATER: 1" in out
    assert "HEATING: 2" in out
    # the citation snapshot reflects the combined heat count (not just HEAT/HOT WATER)
    snapshot = citations.mapping()["S1"]["provenance"]["snapshot"]
    assert snapshot["heat_hot_water"] == 2
    assert snapshot["heat_hot_water_problem_rows"] == 3


async def test_hpd_complaint_total_deduplicates_ids_but_categories_count_problem_rows(monkeypatch):
    complaints = [
        {"complaint_id": "1", "major_category": "HEAT/HOT WATER", "received_date": "2026-06-01"},
        {"complaint_id": "1", "major_category": "PLUMBING", "received_date": "2026-06-02"},
        {"complaint_id": "2", "major_category": "PLUMBING", "received_date": "2026-06-03"},
        {"complaint_id": "", "major_category": "PAINT/PLASTER", "received_date": "2026-06-04"},
    ]
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_lookup(monkeypatch, geopoint=origin, complaints=complaints, violations=[])

    assert "Open HPD complaints: at least 2 total" in out
    assert "1 problem row lacked a complaint ID" in out
    assert "By category (problem rows):" in out
    assert "HEAT/HOT WATER: 1" in out and "PLUMBING: 2" in out and "PAINT/PLASTER: 1" in out
    snapshot = citations.mapping()["S1"]["provenance"]["snapshot"]
    assert snapshot["open_complaints"] == 2
    assert snapshot["complaint_problem_rows"] == 4
    assert snapshot["complaint_rows_missing_id"] == 1


async def test_hpd_building_lookup_marks_1000_row_results_as_lower_bounds(monkeypatch):
    complaints = [
        {"complaint_id": str(i // 2), "major_category": "PLUMBING"} for i in range(1000)
    ]
    violations = [{"class": "B"} for _ in range(1000)]
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_lookup(monkeypatch, geopoint=origin, complaints=complaints, violations=violations)

    assert "Open HPD complaints: at least 500" in out
    assert "Open HPD violations: at least 1,000" in out
    complaint_snapshot = citations.mapping()["S1"]["provenance"]["snapshot"]
    violation_snapshot = citations.mapping()["S2"]["provenance"]["snapshot"]
    assert complaint_snapshot["complaints_truncated"] is True
    assert violation_snapshot["violations_truncated"] is True
    assert complaint_snapshot["open_complaints_lower_bound"] == 500
    assert violation_snapshot["open_violations_lower_bound"] == 1000
    assert "at least 500" in citations.mapping()["S1"]["snippet"]
    assert "at least 1,000" in citations.mapping()["S2"]["snippet"]


# --- 3c. hpd_litigation_lookup: a building's HPD housing-court cases (59kj-x8nc) --------------
#
# Sibling to hpd_building_lookup: whether HPD has taken the landlord to Housing Court, keyed by the
# same BBL. Calls out 'Heat and Hot Water' cases (and which are still pending) plus any finding of
# harassment. Empty is stated plainly ("no cases on record"), never "the landlord is clean".

def _litigation_client(cases: list[dict]) -> httpx.AsyncClient:
    """MockTransport serving the litigations dataset query by its 4x4 id in the path."""
    def handler(request: httpx.Request) -> httpx.Response:
        if LITIGATIONS_ID in request.url.path:
            return httpx.Response(200, json=cases)
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _litigation_tool():
    return next(t for t in get_tools() if t.name == "hpd_litigation_lookup")


async def _run_litigation(monkeypatch, *, geopoint, cases):
    monkeypatch.setattr(housing, "geocode", _fixed_geocode(geopoint))
    citations = CitationRegistry()
    client = _litigation_client(cases)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await _litigation_tool().handler({"address": "617 Courtlandt Ave, Bronx"}, ctx)
    await client.aclose()
    return out, citations


# 617 Courtlandt Ave litigation rows (shape verified live): 4 cases, 2 Heat and Hot Water (1 still
# PENDING), 1 Tenant Action/Harrassment with an 'After Trial' finding of harassment.
_LITIGATIONS = [
    {"bbl": BBL, "boroid": "2", "block": "2411", "lot": "48", "casetype": "Heat and Hot Water",
     "caseopendate": "2026-03-11T00:00:00.000", "casestatus": "PENDING",
     "findingofharassment": "", "respondent": "617 PARTNERS LLC"},
    {"bbl": BBL, "casetype": "Heat and Hot Water", "caseopendate": "2015-03-09T00:00:00.000",
     "casestatus": "CLOSED", "findingofharassment": "", "respondent": "617 COURTLAND AVE CORP"},
    {"bbl": BBL, "casetype": "Tenant Action", "caseopendate": "2026-03-05T00:00:00.000",
     "casestatus": "PENDING", "findingofharassment": "", "respondent": "ALILAH MANAGEMENT"},
    {"bbl": BBL, "casetype": "Tenant Action/Harrassment", "caseopendate": "2009-08-13T00:00:00.000",
     "casestatus": "CLOSED", "findingofharassment": "After Trial", "respondent": "LUIS B FABRE"},
]


async def test_hpd_litigation_lookup_grounds_cases_callouts_and_cites(monkeypatch):
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_litigation(monkeypatch, geopoint=origin, cases=_LITIGATIONS)

    assert LABEL in out and BBL in out
    # 4 cases total; 2 are Heat and Hot Water, 1 of them still pending
    assert "4 total" in out
    assert "2 Heat and Hot Water" in out
    assert "1 currently pending" in out
    # a positive finding of harassment (After Trial) is surfaced, not swallowed
    assert "Findings of harassment: 1" in out
    # case-type breakdown preserves HPD's real (mixed-case) casetype strings, not uppercased
    assert "Heat and Hot Water: 2" in out
    assert "Tenant Action: 1" in out
    # most recent case-open date comes from the rows, never invented
    assert "2026-03-11" in out
    # exactly one DATA citation, addressed to the filtered BBL query on the litigations dataset
    mapping = citations.mapping()
    assert len(mapping) == 1
    cite = mapping["S1"]
    assert cite["kind"] == "DATA"
    assert LITIGATIONS_ID in cite["url"] and BBL in cite["url"]
    assert "{cite:S1}" in out
    # provenance grounded in the actual rows
    assert cite["provenance"]["record_id"] == BBL
    assert cite["provenance"]["snapshot"]["cases"] == 4
    assert cite["provenance"]["snapshot"]["heat_and_hot_water"] == 2
    assert cite["valid_as_of"] == "2026-03-11"


async def test_hpd_litigation_lookup_empty_states_no_cases_not_clean(monkeypatch):
    origin = GeoPoint(lat=40.8195, lon=-73.9160, label=LABEL, bbl=BBL)
    out, citations = await _run_litigation(monkeypatch, geopoint=origin, cases=[])
    low = out.lower()
    assert "no hpd housing-court" in low          # states the absence plainly
    # never spun into "the landlord is clean / problem-free"
    assert "problem-free" not in low and "clean" not in low
    # still grounded: a DATA citation to the (empty) filtered query
    mapping = citations.mapping()
    assert len(mapping) == 1
    assert mapping["S1"]["kind"] == "DATA"
    assert LITIGATIONS_ID in mapping["S1"]["url"]
    assert "{cite:S1}" in out


async def test_hpd_litigation_lookup_abstains_without_bbl(monkeypatch):
    no_bbl = GeoPoint(lat=40.8195, lon=-73.9160, label="ZIP 10451 area", bbl="")
    out, citations = await _run_litigation(monkeypatch, geopoint=no_bbl, cases=_LITIGATIONS)
    low = out.lower()
    assert "couldn't" in low or "could not" in low
    assert "street address" in low               # asks for a specific street address
    assert "{cite:" not in out
    assert len(citations) == 0


# --- 4. housing_guidance: static-but-official facts, each cited to nyc.gov ---
#
# These are offline (no network): the tool bakes the facts + source URLs in and only touches the
# citation registry. They lock in what the eval's tool_sanity / attribution / faithfulness checks
# rely on — a grounding tool call and a citation whose snippet is backed by the returned fact.

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _guidance_tool():
    return next(t for t in get_tools() if t.name == "housing_guidance")


async def _run_guidance(topic: str):
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]))
    out = await _guidance_tool().handler({"topic": topic}, ctx)
    return out, citations


async def test_guidance_right_to_counsel_grounds_legal_help_and_cites():
    out, citations = await _run_guidance("right_to_counsel")
    assert "free legal help" in out.lower()
    assert "718-557-1379" in out          # Housing Court Answers, from the official page
    assert "311" in out
    assert "1118 Grand Concourse" not in out
    assert "718-466-3022" not in out
    # Generic citywide legal help must not inject a Bronx-specific destination
    mapping = citations.mapping()
    assert len(mapping) == 1
    cite = mapping["S1"]
    assert cite["kind"] == "DOC"
    assert cite["url"] == "https://www.nyc.gov/site/hra/help/legal-services-for-tenants.page"
    assert "{cite:S1}" in out             # the fact carries its citation inline


async def test_guidance_bronx_housing_court_returns_direct_distinct_locations():
    out, citations = await _run_guidance("bronx_housing_court")

    assert "1118 Grand Concourse" in out
    assert "718-466-3022" in out
    assert "851 Grand Concourse" in out
    assert "718-466-3025" in out
    mapping = citations.mapping()
    assert len(mapping) == 1
    assert mapping["S1"]["url"].endswith("bronx-county-housing-court-directory")
    assert mapping["S1"]["valid_as_of"] == "2026-08-07"
    assert "{cite:S1}" in out


async def test_guidance_no_heat_grounds_standard_code_ladder_and_cites():
    out, citations = await _run_guidance("no_heat")
    low = out.lower()
    assert "october 1" in low and "may 31" in low   # heat season, from the HPD page
    assert "current applicability" in low
    assert "68" in out and "62" in out and "55" in out and "120" in out
    assert "311" in out                              # how to file
    # the exact Housing Maintenance Code sections, grounded in the statute, not just the explainer
    assert "27-2029" in out and "27-2031" in out
    # the escalation ladder rungs are stated (immediately hazardous violation → Housing Court)
    assert "class C" in out or "immediately hazardous" in out
    assert "housing court" in low
    # three DOC citations: HPD, section 27-2029, and section 27-2031
    mapping = citations.mapping()
    assert len(mapping) == 3
    assert mapping["S1"]["kind"] == "DOC"
    assert mapping["S1"]["url"].endswith("heat-and-hot-water-information.page")
    assert mapping["S2"]["kind"] == "DOC"
    assert mapping["S2"]["url"] == "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-60410"
    assert "27-2029" in mapping["S2"]["title"]
    assert mapping["S3"]["url"] == "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-236495"
    assert "27-2031" in mapping["S3"]["title"]
    assert "{cite:S1}" in out and "{cite:S2}" in out and "{cite:S3}" in out


def test_heat_season_status_states_current_applicability():
    assert "currently in effect" in housing._heat_season_status(date(2026, 1, 15))
    assert "not currently in effect" in housing._heat_season_status(date(2026, 7, 27))
    assert "currently in effect" in housing._heat_season_status(date(2026, 11, 15))


async def test_guidance_no_heat_evidence_supports_its_measured_units():
    _out, citations = await _run_guidance("no_heat")
    answer = (
        "During heat season, apartments must be at least 68°F when it is below 55°F "
        "outside during the day, at least 62°F overnight, and hot water must be at "
        "least 120°F year-round. {cite:S1}"
    )

    verdict = check_grounding(answer, citations.mapping())

    assert verdict is not None and not verdict.blocking


async def test_guidance_no_water_grounds_cold_water_service_not_hot_standard():
    # F070: "no cold water" / "no water at all" is a water-service problem, not the hot-water
    # 120°F standard. The topic frames it correctly (owner must provide cold water, file a
    # complaint, Housing Court) and NEVER returns the hot-water temperature standard.
    out, citations = await _run_guidance("no_water")
    low = out.lower()
    assert "cold water" in low
    assert "120" not in out                        # the F070 inversion must not reappear
    assert "311" in out
    mapping = citations.mapping()
    assert len(mapping) == 1
    cite = mapping["S1"]
    assert cite["kind"] == "DOC"
    assert "tenants-rights-and-responsibilities" in cite["url"]
    assert "{cite:S1}" in out


async def test_guidance_no_heat_still_returns_hot_water_standard():
    # F070 fence: the hot-water/heat topic keeps the existing 120°F heat-season standard;
    # only cold-water/no-water is reframed, not heat.
    out, _ = await _run_guidance("no_heat")
    assert "120" in out and "68" in out


async def test_guidance_cold_water_freetext_does_not_return_hot_water_standard():
    # F070: the ambiguous "cold" keyword no longer inverts a cold-water complaint into the
    # hot-water standard. Ungrounded free text abstains rather than mis-framing.
    out, _ = await _run_guidance("my building has no cold water today")
    assert "120" not in out


async def test_guidance_shelter_grounds_both_intakes_and_cites_each():
    out, citations = await _run_guidance("shelter")
    # families → PATH, single adults → current men's site / Franklin
    # F161: the men's intake relocated 2026-08-01; assert whichever site is current
    # Hardcoding it made this expire at midnight while the helper stayed correct
    assert "PATH" in out and "151 East 151st Street" in out and "718-503-6400" in out
    assert _single_adult_men_intake() in out and "Franklin Shelter" in out
    assert "Help Women's Center" in out and "114 Snediker Avenue" in out
    assert "333 Bowery" in out
    # three DOC citations, one per household/intake type, each cited inline
    mapping = citations.mapping()
    assert len(mapping) == 3
    assert {c["kind"] for c in mapping.values()} == {"DOC"}
    urls = {c["url"] for c in mapping.values()}
    assert any("families-with-children-applying" in u for u in urls)
    assert any("adult-families-applying" in u for u in urls)
    assert any("single-adults-applying" in u for u in urls)
    assert "{cite:S1}" in out and "{cite:S2}" in out and "{cite:S3}" in out


def test_shelter_men_intake_helper_handles_the_official_transition_dates():
    assert "30th Street" in _single_adult_men_intake(date(2026, 7, 31))
    assert "8 East 3rd Street" in _single_adult_men_intake(date(2026, 8, 1))


async def test_guidance_maps_free_text_to_topic():
    # the model may pass the user's words instead of a canonical key — they map to a topic.
    heat_out, _ = await _run_guidance("my landlord shut off the heat")
    assert "october 1" in heat_out.lower()
    lawyer_out, _ = await _run_guidance("I need a lawyer for my eviction case")
    assert "free legal help" in lawyer_out.lower()
    shelter_out, _ = await _run_guidance("we have nowhere to stay tonight")
    assert "PATH" in shelter_out
    adult_family_out, _ = await _run_guidance("we are an adult family and need intake")
    assert "333 Bowery" in adult_family_out


async def test_guidance_source_of_income_grounds_protection_and_cites():
    out, citations = await _run_guidance("source_of_income")
    low = out.lower()
    # the NYC protection is AFFIRMED (never hedged), with the complaint phone + the statute section
    assert "source-of-income" in low or "source of income" in low
    assert "illegal" in low
    assert "212-416-0197" in out                     # NYC Commission on Human Rights
    assert "8-107(5)" in out                          # the NYC Human Rights Law section
    assert "Section 8" in out and "CityFHEPS" in out
    # two DOC citations: the CCHR explainer (S1) + the amlegal statute (S2), each cited inline
    mapping = citations.mapping()
    assert len(mapping) == 2
    assert {c["kind"] for c in mapping.values()} == {"DOC"}
    urls = {c["url"] for c in mapping.values()}
    assert any("cchr" in u for u in urls)
    assert any("amlegal" in u for u in urls)
    assert "{cite:S1}" in out and "{cite:S2}" in out


async def test_guidance_maps_voucher_refusal_free_text_to_source_of_income():
    # "won't take my voucher"-style free text resolves to the source_of_income topic
    out, _ = await _run_guidance("my landlord won't take my voucher")
    assert "212-416-0197" in out                      # resolved to source_of_income
    assert "illegal" in out.lower()


async def test_guidance_unknown_topic_abstains_without_citation():
    out, citations = await _run_guidance("rent freeze eligibility")
    assert len(citations) == 0             # nothing grounded → nothing cited
    assert "311" in out                    # routes the user onward
    assert "free eviction lawyer" not in out
    assert "{cite:" not in out


async def test_guidance_citation_snippets_are_backed_by_the_returned_facts():
    """Mirror the eval's faithfulness check offline: every citation snippet's tokens are ≥60%
    covered by the tool's own output, so a DOC citation can never outrun the fact it cites."""
    for topic in ("right_to_counsel", "no_heat", "no_water", "shelter", "source_of_income"):
        out, citations = await _run_guidance(topic)
        haystack = set(_TOKEN_RE.findall(out.lower()))
        for cid, c in citations.mapping().items():
            tokens = [t for t in _TOKEN_RE.findall(c["snippet"].lower()) if len(t) > 1]
            overlap = sum(1 for t in tokens if t in haystack) / len(tokens)
            assert overlap >= 0.6, f"{topic}/{cid} snippet under-backed by output ({overlap:.0%})"


# --- 5. the shipped module stays valid -------------------------------------

def test_housing_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "housing"), None)
    assert module is not None
    assert module.category == "housing"
    assert "Do not describe this as recent reports" in module.prompt
    assert "free legal representation or advice" in module.prompt
    assert "a free lawyer" not in module.prompt
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "hpd_building_lookup" in tool_names
    assert "hpd_litigation_lookup" in tool_names
    assert "housing_guidance" in tool_names
    guidance = next(t for t in get_tools() if t.name == "housing_guidance")
    assert "free legal representation or advice" in guidance.description
    assert "the FREE lawyer" not in guidance.description

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "housing"]
    assert cases, "housing should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
    right_to_counsel = next(c for c in cases if c.id == "housing_right_to_counsel")
    assert "free legal representation or advice" in right_to_counsel.notes
    assert "free lawyer" not in right_to_counsel.notes
    # the routing cases now expect the grounding tool + a citation (cite-or-abstain)
    routing = {c.id: c for c in cases if c.id in {
        "housing_right_to_counsel", "housing_no_heat", "housing_shelter_family",
        "housing_shelter_single_adult"}}
    assert len(routing) == 4
    for case in routing.values():
        assert "housing_guidance" in case.expect_tools
        assert case.invariants.get("must_cite_if_asserting")
