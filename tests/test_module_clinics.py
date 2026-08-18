"""Offline tests for the clinics module (FQHC + NYC Care).

Grounded in HRSA's ArcGIS FQHC layer + a bundled NYC Care/H+H seed, but every HTTP call is
mocked/injected, no live HRSA or geocoder call. Covers: FQHC record mapping + county->borough,
NYC Care seed loading, merge + rank by distance across both classes, per-class DATA (facility) and
DOC (program) citations, the ANTI-HALLUCINATION guarantee that eligibility text is grounded to the
program page (never a per-row field / the model), abstention on geocode failure, and low-confidence
clarification. Mirrors tests/test_module_food_pantries.py.
"""
from __future__ import annotations

import httpx
from pydantic import BaseModel

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.arcgis import ArcGISQueryPage
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.clinics import tools as clinics
from heynyc.modules.clinics.tools import (
    CLASS_FQHC,
    CLASS_GUARANTEE,
    CLASS_NYC_CARE,
    _fqhc_from_record,
    _load_nyc_care_seed,
    get_tools,
)

HRSA_HOST = "gisportal.hrsa.gov"
GEOSEARCH_HOST = "geosearch.planninglabs.nyc"


# --- pure helpers ----------------------------------------------------------

def test_fqhc_from_record_maps_fields_and_borough():
    rec = {"SITE_NM": "Test Health Center", "SITE_ADDRESS": "1 Main St", "SITE_CITY": "Bronx",
           "SITE_ZIP_CD": "10460-1234", "SITE_PHONE_NUM": "718-555-0100", "SITE_URL": "www.x.org",
           "COUNTY_NM": "Kings", "OBJECTID": 42, "lat": 40.75, "lon": -73.99}
    clinic = _fqhc_from_record(rec)
    assert clinic is not None
    assert clinic.klass == CLASS_FQHC
    assert clinic.name == "Test Health Center"
    assert clinic.borough == "Brooklyn"                 # COUNTY_NM 'Kings' -> borough
    assert clinic.address == "1 Main St, Bronx 10460"   # zip truncated to 5
    assert clinic.phone == "718-555-0100"
    assert clinic.record_id == "42"


def test_fqhc_from_record_drops_bad_coords_and_na_url():
    assert _fqhc_from_record({"SITE_NM": "No Coords", "OBJECTID": 1}) is None
    clinic = _fqhc_from_record({"SITE_NM": "X", "SITE_URL": "N/A", "COUNTY_NM": "Queens",
                                "OBJECTID": 2, "lat": 40.7, "lon": -73.8})
    assert clinic.url == ""                             # literal 'N/A' scrubbed to blank


def test_nyc_care_seed_loads_and_is_grounded_in_nyc():
    """The shipped seed loads, is non-trivial, and every row has NYC-ish coordinates + a class tag."""
    catalog = _load_nyc_care_seed()
    seed = catalog.clinics
    assert catalog.complete is True
    assert catalog.returned_count == len(seed)
    assert len(seed) >= 35                               # 11 hospitals + ~29 Gotham centers
    assert all(c.klass == CLASS_NYC_CARE for c in seed)
    assert any("Gotham Health" in c.name for c in seed)
    assert any("Bellevue" in c.name for c in seed)
    for c in seed:
        assert 40.4 < c.lat < 41.0 and -74.3 < c.lon < -73.6   # inside the NYC bbox


def test_missing_seed_degrades_to_empty(tmp_path):
    catalog = _load_nyc_care_seed(tmp_path / "nope.tsv")
    assert catalog.clinics == []
    assert catalog.complete is False
    assert catalog.error == "file_unavailable"


# --- the tool handler (mocked ArcGIS + injected seed) ----------------------

def _geojson(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _fqhc_feature(lon, lat, **props) -> dict:
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props}


def _routed_client(features) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if GEOSEARCH_HOST in host:
            return httpx.Response(200, json={"features": [
                {"geometry": {"coordinates": [-73.9900, 40.7500]},
                 "properties": {"label": "Origin, Manhattan"}}]})
        if HRSA_HOST in host:
            return httpx.Response(200, json=_geojson(*features))
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _seed_clinic(monkeypatch, *clinics_):
    """Replace the bundled seed loader with a fixed in-memory NYC Care list."""
    monkeypatch.setattr(
        clinics,
        "_load_nyc_care_seed",
        lambda *a, **k: clinics._NycCareCatalog(
            clinics=list(clinics_),
            returned_count=len(clinics_),
            complete=True,
        ),
    )


def _nyc_care(name, lat, lon, phone="") -> clinics.Clinic:
    return clinics.Clinic(name=name, lat=lat, lon=lon, address=f"{name} addr", borough="Manhattan",
                          phone=phone, url="", klass=CLASS_NYC_CARE, record_id=name,
                          valid_as_of="2026-07-04", raw={"name": name})


async def test_find_clinic_merges_ranks_and_attaches_per_class_citations(monkeypatch):
    # A near NYC Care site (should rank #1) + two FQHCs at varying distance + one bad-coords FQHC.
    _seed_clinic(monkeypatch, _nyc_care("H+H Test Hospital", 40.7501, -73.9901, phone="212-555-9999"))
    features = [
        _fqhc_feature(-73.9600, 40.8000, SITE_NM="Far FQHC", SITE_ADDRESS="9 Far St",
                      SITE_CITY="New York", SITE_ZIP_CD="10027", SITE_PHONE_NUM="212-555-0001",
                      COUNTY_NM="New York", OBJECTID=1),
        _fqhc_feature(-73.9910, 40.7510, SITE_NM="Close FQHC", SITE_ADDRESS="2 Near Ave",
                      SITE_CITY="New York", SITE_ZIP_CD="10001", SITE_PHONE_NUM="212-555-0002",
                      COUNTY_NM="New York", OBJECTID=2),
        _fqhc_feature(None, None, SITE_NM="No Coords FQHC", COUNTY_NM="New York", OBJECTID=3),
    ]
    citations = CitationRegistry()
    client = _routed_client(features)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    assert isinstance(out, BaseModel)
    assert out.outcome == "success"
    assert [record.organization.name for record in out.clinics] == [
        "H+H Test Hospital",
        "Close FQHC",
        "Far FQHC",
    ]
    assert out.origin.precision == "approximate"
    assert out.sources.hrsa.status == "ok"
    assert out.sources.hrsa.returned_count == 3
    assert out.sources.hrsa.usable_count == 2
    assert out.sources.hrsa.complete is True
    assert out.sources.hrsa.fetched_at is not None
    assert out.sources.nyc_care.status == "ok"
    assert all(
        record.valid_as_of is None
        for record in out.clinics
        if record.service.program_class == CLASS_FQHC
    )
    assert all(record.service.hours is None for record in out.clinics)
    assert {fact.program_class for fact in out.program_guarantees} == {CLASS_FQHC, CLASS_NYC_CARE}

    mapping = citations.mapping()
    # Facility DATA citations: FQHC -> HRSA row permalink (OBJECTID), NYC Care -> H+H locations page.
    data = {cid: c for cid, c in mapping.items() if c["kind"] == "DATA"}
    assert any("hrsa.gov" in c["url"].lower() and "objectid" in c["url"].lower() for c in data.values())
    assert any("nychealthandhospitals.org" in c["url"].lower() for c in data.values())
    # Program DOC citations: exactly one per class present, grounded to the official program page.
    docs = {cid: c for cid, c in mapping.items() if c["kind"] == "DOC"}
    doc_urls = {c["url"] for c in docs.values()}
    assert "https://www.hrsa.gov/get-health-care" in doc_urls
    assert "https://access.nyc.gov/programs/nyc-care/" in doc_urls
    assert len(docs) == 2                                # deduped: one DOC per class, not per site


async def test_eligibility_text_is_grounded_not_from_a_row_field(monkeypatch):
    """ANTI-HALLUCINATION: the eligibility/immigration framing is the CLASS guarantee body, cited to
    the program DOC, never a per-row field, even if the record carries a bogus cost/eligibility one."""
    _seed_clinic(monkeypatch)  # no NYC Care sites → only the FQHC class block
    # Poison the record with fake per-row cost/eligibility fields; they must NOT appear in output.
    features = [_fqhc_feature(-73.9910, 40.7510, SITE_NM="Poison FQHC", SITE_ADDRESS="2 Near Ave",
                              SITE_CITY="New York", SITE_ZIP_CD="10001", COUNTY_NM="New York",
                              OBJECTID=7, FEE="$500 per visit", ELIGIBILITY="citizens only",
                              COST="not free")]
    citations = CitationRegistry()
    client = _routed_client(features)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    serialized = out.model_dump_json()
    assert "$500" not in serialized and "citizens only" not in serialized and "not free" not in serialized
    assert any("sliding fee scale" in fact.verified_fact for fact in out.program_guarantees)
    assert CLASS_GUARANTEE[CLASS_FQHC].doc_url in citations.mapping()["S2"]["url"] \
        or any("hrsa.gov/get-health-care" in c["url"] for c in citations.mapping().values())


async def test_find_clinic_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr(clinics, "geocode", fail)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": []})))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert out.outcome == "location_not_found"
    assert out.clinics == []
    assert {route.name for route in out.fallback_routes} == {
        "NYC Care",
        "HRSA Find a Health Center",
    }


async def test_find_clinic_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr(clinics, "geocode", ambiguous)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert out.outcome == "location_ambiguous"
    assert out.clinics == []
    assert out.origin.low_confidence is True


async def test_find_clinic_degrades_to_seed_when_hrsa_down(monkeypatch):
    """If HRSA is unreachable, the tool still serves the bundled NYC Care seed (never fully abstains
    when the safety-net answer still stands)."""
    _seed_clinic(monkeypatch, _nyc_care("H+H Fallback", 40.7502, -73.9902))

    def handler(request: httpx.Request) -> httpx.Response:
        if GEOSEARCH_HOST in request.url.host:
            return httpx.Response(200, json={"features": [
                {"geometry": {"coordinates": [-73.9900, 40.7500]}, "properties": {"label": "Origin"}}]})
        return httpx.Response(503)                        # HRSA down
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    assert out.outcome == "degraded"
    assert [record.organization.name for record in out.clinics] == ["H+H Fallback"]
    assert out.sources.hrsa.status == "unavailable"
    assert out.sources.nyc_care.status == "ok"


async def test_hrsa_query_paginates_without_losing_provider_metadata(monkeypatch):
    calls = []

    async def query(*args, result_offset=0, pagination_token=None, **kwargs):
        calls.append((result_offset, pagination_token))
        if pagination_token is None:
            return ArcGISQueryPage(
                records=[{"OBJECTID": 1}],
                exceeded_transfer_limit=True,
                next_offset=None,
                pagination_token="page-1",
            )
        assert pagination_token == "page-1"
        return ArcGISQueryPage(
            records=[{"OBJECTID": 2}],
            exceeded_transfer_limit=False,
            next_offset=None,
            pagination_token="page-2",
        )

    monkeypatch.setattr(clinics, "query_feature_service_page", query)
    page = await clinics._query_hrsa(httpx.AsyncClient())

    assert calls == [(0, None), (0, "page-1")]
    assert [record["OBJECTID"] for record in page.records] == [1, 2]
    assert page.complete is True
    assert page.pages_fetched == 2
    assert page.pagination_token == "page-2"


async def test_hrsa_query_preserves_partial_page_on_later_failure(monkeypatch):
    async def query(*args, pagination_token=None, **kwargs):
        if pagination_token is None:
            return ArcGISQueryPage(
                records=[{"OBJECTID": 1}],
                exceeded_transfer_limit=True,
                next_offset=2000,
                pagination_token="page-1",
            )
        raise httpx.ReadTimeout("later page failed")

    monkeypatch.setattr(clinics, "query_feature_service_page", query)
    page = await clinics._query_hrsa(httpx.AsyncClient())

    assert page.records == [{"OBJECTID": 1}]
    assert page.complete is False
    assert page.pages_fetched == 1
    assert page.next_offset is None
    assert page.pagination_token == "page-1"
    assert page.error == "transport_error"


async def test_find_clinic_does_not_rank_incomplete_hrsa_rows(monkeypatch):
    _seed_clinic(monkeypatch, _nyc_care("Complete H+H", 40.7502, -73.9902))

    async def partial(*args, **kwargs):
        return clinics._HrsaQueryResult(
            records=[{"OBJECTID": 1, "SITE_NM": "Partial FQHC", "lat": 40.75, "lon": -73.99}],
            pages_fetched=1,
            complete=False,
            next_offset=2000,
            error="transport_error",
        )

    monkeypatch.setattr(clinics, "_query_hrsa", partial)
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    assert out.outcome == "degraded"
    assert [record.organization.name for record in out.clinics] == ["Complete H+H"]
    assert out.sources.hrsa.status == "partial"
    assert out.sources.hrsa.returned_count == 1
    assert out.sources.hrsa.complete is False


# --- get_health_coverage_guidance: official coverage facts, each cited (no network) -----

def _coverage_tool():
    return next(t for t in get_tools() if t.name == "get_health_coverage_guidance")


async def _run_coverage(topic: str):
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]))
    out = await _coverage_tool().handler({"topic": topic}, ctx)
    return out, citations


async def test_health_coverage_emergency_medicaid_grounds_body_and_cites():
    out, citations = await _run_coverage("emergency_medicaid")
    assert "Emergency Medicaid" in out
    assert "regardless of immigration status" in out
    assert "emergency labor and delivery and kidney dialysis" in out
    assert "HRA Medicaid Helpline" in out
    assert "(888) 692-6116" in out
    assert "Do not add urgency, timing, or an intake script" in out
    assert "Keep each fact's citation only on claims from that fact" in out
    mapping = citations.mapping()
    assert len(mapping) == 2
    cite = mapping["S1"]
    assert cite["kind"] == "DOC"
    assert "health.ny.gov" in cite["url"]
    assert "{cite:S1}" in out
    assert "emergency labor and delivery" in cite["provenance"]["snapshot"]["verified_fact"]
    application = mapping["S2"]
    assert application["url"].endswith("/medicaid/how_do_i_apply.htm")
    assert "HRA Medicaid Helpline" in application["provenance"]["snapshot"]["verified_fact"]
    assert "{cite:S2}" in out
    assert "public charge" not in out.lower()
    assert "ActionNYC" not in out


async def test_health_coverage_tourist_emergency_care_separates_treatment_from_billing():
    out, citations = await _run_coverage("emergency_care")
    assert "cannot deny an emergency screening" in out
    assert "does not mean the care is free" in out
    assert "Emergency Medicaid eligibility is a separate question" in out
    mapping = citations.mapping()
    assert len(mapping) == 1
    assert mapping["S1"]["url"] == "https://www.cms.gov/priorities/your-patient-rights/emergency-room-rights"
    assert "{cite:S1}" in out


async def test_health_coverage_nyc_care_grounds_body_and_cites():
    out, citations = await _run_coverage("nyc_care")
    assert "NYC Care" in out
    assert "646-NYC-CARE (646-692-2273)" in out
    assert "doesn't ask about immigration status" in out
    mapping = citations.mapping()
    assert len(mapping) == 1
    assert mapping["S1"]["kind"] == "DOC"
    assert mapping["S1"]["url"] == "https://access.nyc.gov/programs/nyc-care/"
    assert "{cite:S1}" in out
    assert "ActionNYC" not in out and "public charge" not in out.lower()


async def test_health_coverage_public_charge_grounds_body_and_cites():
    out, citations = await _run_coverage("public_charge")
    assert "do not count against you" in out
    assert "final rule" in out
    assert "September 18, 2026" in out
    assert "means-tested public benefits" in out
    assert "nothing has changed" not in out
    assert "800-354-0365" in out
    mapping = citations.mapping()
    assert len(mapping) == 2
    assert mapping["S1"]["kind"] == "DOC"
    assert "nyc.gov/site/immigrants" in mapping["S1"]["url"]
    assert "800-354-0365" in mapping["S1"]["snippet"]
    assert "{cite:S1}" in out
    assert mapping["S2"]["kind"] == "DOC"
    assert "federalregister.gov/documents/2026/07/20/2026-14539" in mapping["S2"]["url"]
    assert "{cite:S2}" in out


async def test_health_coverage_does_not_interpret_free_text_tool_arguments():
    out, citations = await _run_coverage(
        "I'm undocumented and pregnant, how do I pay for the delivery"
    )

    assert "Emergency Medicaid" not in out
    assert len(citations) == 0


async def test_health_coverage_unknown_topic_abstains_without_citation():
    out, citations = await _run_coverage("how much is a dental cleaning")
    assert len(citations) == 0
    assert "{cite:" not in out
    assert "646-NYC-CARE" in out or "311" in out     # routes the user onward


# --- the shipped module stays valid ---------------------------------------

def test_clinics_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "clinics"), None)
    assert module is not None
    assert module.category == "health"
    discovery_summary = " ".join(module.description.split())[:140].lower()
    assert "public charge" in discovery_summary
    assert "green card" in discovery_summary
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "find_clinics" in tool_names
    assert "get_health_coverage_guidance" in tool_names
    # the FQHC dataset binding is declared + discoverable
    assert "fqhc_site" in registry.dataset_bindings()

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "clinics"]
    assert cases, "clinics should ship eval cases"
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)


def test_health_coverage_tool_separates_delivery_coverage_from_nyc_care():
    description = _coverage_tool().description.lower()

    assert "delivery payment" in description
    assert "use `emergency_medicaid`" in description
    assert "ongoing non-emergency care" in description
    assert "do not also call `nyc_care`" in description


def test_clinics_module_scopes_call_ahead_disclaimer_to_location_results():
    registry = Registry.discover(config.MODULES_DIR)
    module = next(module for module in registry.modules if module.name == "clinics")
    prompt = module.prompt.lower()

    assert "after `find_clinics` returns locations" in prompt
    assert "do not append this location disclaimer to coverage-only answers" in prompt
