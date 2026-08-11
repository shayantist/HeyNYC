"""Offline tests for the food_pantries module.

Grounded in the city's FoodHelp ArcGIS backend, but every HTTP call is mocked/injected —
no live ArcGIS or geocoder call. Covers: ranking by distance, open-now computation from the
structured fp_<day>_open*/close* hours, dietary/access flags, the directions link, a grounded
DATA citation, and abstention when geocoding fails.
"""
from __future__ import annotations

from datetime import date, datetime

import httpx
import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.grounding import check_grounding
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.food_pantries import tools as fp
from heynyc.modules.food_pantries.tools import (
    _DAYS,
    _flags,
    _open_now,
    _parse_time,
    _to_pantry,
    _valid_as_of,
    directions_link,
    get_tools,
)


class _Noon(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 7, 17, 12, 0, tzinfo=tz)


# --- pure helpers ----------------------------------------------------------

def test_parse_time_handles_common_formats():
    assert _parse_time("9:00 AM") == 9 * 60
    assert _parse_time("12:00 PM") == 12 * 60
    assert _parse_time("12:00 AM") == 0
    assert _parse_time("5:30 PM") == 17 * 60 + 30
    assert _parse_time("17:30") == 17 * 60 + 30
    assert _parse_time("0900") == 9 * 60
    assert _parse_time(None) is None
    assert _parse_time("") is None
    assert _parse_time("NULL") is None


def test_directions_link_is_google_maps_dir():
    link = directions_link(40.7484, -73.9857)
    assert link == "https://www.google.com/maps/dir/?api=1&destination=40.74840,-73.98570"


def test_flags_reads_dietary_access_from_type_fp():
    assert _flags(_pantry(type_fp="FPH")) == ["Halal"]
    assert _flags(_pantry(type_fp="FPHA")) == ["HIV Customers"]
    assert _flags(_pantry(type_fp="FPK")) == ["Kosher"]
    assert _flags(_pantry(type_fp="FPM")) == ["Mobile"]
    assert _flags(_pantry(type_fp="FP")) == []  # plain pantry → no special flag
    assert _flags(_pantry(type_fp="", type_sk="SKK")) == ["Kosher"]  # soup kitchen domain


def _pantry(**over):
    base = {"program": "Test Pantry", "lat": 40.75, "lon": -73.99, "type_fp": "FP"}
    base.update(over)
    return _to_pantry(base)


def _hours_record(day: str, open1: str, close1: str, **extra) -> dict:
    rec = {"program_type": "FP", f"fp_{day}_open1": open1, f"fp_{day}_close1": close1}
    rec.update(extra)
    return rec


def test_open_now_true_within_hours():
    now = datetime(2026, 7, 1, 12, 0)              # noon
    day = _DAYS[now.weekday()]
    rec = _hours_record(day, "9:00 AM", "5:00 PM")
    assert _open_now(rec, now) is True


def test_open_now_reads_the_source_third_daily_slot():
    now = datetime(2026, 7, 1, 20, 0)
    day = _DAYS[now.weekday()]
    rec = _hours_record(
        day,
        "9:00 AM",
        "10:00 AM",
        **{f"fp_{day}_open3": "7:00 PM", f"fp_{day}_close3": "9:00 PM"},
    )
    assert _open_now(rec, now) is True


def test_open_now_false_outside_hours():
    now = datetime(2026, 7, 1, 20, 0)              # 8pm, after close
    day = _DAYS[now.weekday()]
    rec = _hours_record(day, "9:00 AM", "5:00 PM")
    assert _open_now(rec, now) is False


def test_open_now_false_when_closed_today_but_open_other_days():
    now = datetime(2026, 7, 1, 12, 0)
    today = _DAYS[now.weekday()]
    other = _DAYS[(now.weekday() + 1) % 7]
    rec = _hours_record(other, "9:00 AM", "5:00 PM")  # hours exist, but not today
    assert today not in rec
    assert _open_now(rec, now) is False


def test_open_now_none_when_no_hours_at_all():
    now = datetime(2026, 7, 1, 12, 0)
    assert _open_now({"program_type": "FP"}, now) is None  # honest unknown, never a guess


def test_f194_open_now_rejects_foodhelp_cross_midnight_hours():
    late = datetime(2026, 7, 1, 23, 30)
    late_day = _DAYS[late.weekday()]
    overnight = _hours_record(
        late_day,
        "9:00 AM",
        "5:00 AM",
        fp_days_orig="TUE-FRI",
        fp_hours_orig="11AM-3PM",
    )
    assert _open_now(overnight, late) is None

    early = datetime(2026, 7, 2, 1, 30)
    assert _open_now(overnight, early) is None
    assert fp._listed_hours(overnight, late.weekday()) == ""


def test_source_date_preserves_valid_values_and_rejects_invalid_values():
    assert _valid_as_of({"EditDate": "2025-11-05T10:30:00Z"}) == "2025-11-05"
    assert _valid_as_of({"EditDate": 1762300800000}) == "2025-11-05"
    assert _valid_as_of({"EditDate": "not-a-date"}) == ""


# --- the tool handler ------------------------------------------------------

FOODHELP_HOST = "services6.arcgis.com"
GEOSEARCH_HOST = "geosearch.planninglabs.nyc"


def _geojson(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _pantry_feature(lon, lat, **props) -> dict:
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}


def _routed_client(features) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if GEOSEARCH_HOST in host:
            return httpx.Response(200, json={"features": [
                {"geometry": {"coordinates": [-73.9900, 40.7500]},
                 "properties": {"label": "Origin, Manhattan"}}]})
        if FOODHELP_HOST in host:
            return httpx.Response(200, json=_geojson(*features))
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_nearest_food_pantry_rejects_model_invented_origin(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("invented location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="My SNAP stopped and I need food today.",
        user_history="My SNAP stopped and I need food today.",
        user_turns=("My SNAP stopped and I need food today.",),
    )

    out = await get_tools()[0].handler({"near": "Lower East Side"}, ctx)

    assert "where" in out.lower()
    assert "address or neighborhood" in out.lower()
    assert "call 311" in out.lower()
    assert "finder.nyc.gov/foodhelp" in out


async def test_urgent_food_requires_the_residents_service_window(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("incomplete urgent request reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="I need food tonight near Union Square.",
        user_turns=("I need food tonight near Union Square.",),
    )

    out = await get_tools()[0].handler(
        {"near": "Union Square", "urgent": True},
        ctx,
    )

    assert "requires service_window_start and service_window_end" in out


@pytest.mark.parametrize(
    ("near", "query", "history"),
    [
        ("Upper East Side", "I need food today.", "I am in East Harlem.\nI need food today."),
        ("Brooklyn", "I need food today.", "I used to live in Brooklyn.\nI am in Queens.\nI need food today."),
    ],
)
async def test_nearest_food_pantry_rejects_partial_or_stale_origins(
    monkeypatch, near, query, history,
):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("unsupported location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query=query, user_history=history,
        user_turns=tuple(history.splitlines()),
    )

    out = await get_tools()[0].handler({"near": near}, ctx)

    assert "proposed search origin was not supplied" in out


# F159: previously asserted the opposite, conflating two independent properties
# Anti-hallucination (did the RESIDENT author it) is covered by rejects_model_invented_origin
# Staleness is covered by the past-location and negation tests
# "Current message only" proxied both, and the side effect was amnesia
async def test_nearest_food_pantry_uses_a_location_the_resident_gave_in_a_prior_turn(monkeypatch):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="Which one is open now?",
        user_history="I am near Jackson Heights.\nWhich one is open now?",
        user_turns=("I am near Jackson Heights.", "Which one is open now?"),
    )

    await get_tools()[0].handler({"near": "Jackson Heights"}, ctx)

    assert seen == ["Jackson Heights"]


async def test_nearest_food_pantry_rejects_past_location_in_current_turn(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("stale location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    query = "I used to live in Brooklyn, but I am now in Queens."
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query=query, user_turns=(query,),
    )

    out = await get_tools()[0].handler({"near": "Brooklyn"}, ctx)

    assert "proposed search origin was not supplied" in out


async def test_nearest_food_pantry_rejects_negated_current_origin(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("negated location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="I am not in Brooklyn.",
        user_turns=("I am not in Brooklyn.",),
    )

    out = await get_tools()[0].handler({"near": "Brooklyn"}, ctx)

    assert "proposed search origin was not supplied" in out


@pytest.mark.parametrize("query", [
    "I don't live in Brooklyn.",
    "I am not located anywhere near Brooklyn.",
    "Brooklyn is not where I live.",
])
async def test_nearest_food_pantry_rejects_extended_negation(monkeypatch, query):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("negated location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query=query, user_turns=(query,),
    )

    out = await get_tools()[0].handler({"near": "Brooklyn"}, ctx)

    assert "proposed search origin was not supplied" in out


async def test_nearest_food_pantry_preserves_resident_address_abbreviation(monkeypatch):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    query = "I am near 123 Main St, Brooklyn."
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query=query, user_turns=(query,),
    )

    await get_tools()[0].handler({"near": "123 Main Street, Brooklyn"}, ctx)

    assert seen == ["123 Main St, Brooklyn"]


async def test_nearest_food_pantry_accepts_city_qualifiers_added_to_resident_landmark(monkeypatch):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="Is there a food pantry open right now near Union Square?",
        user_turns=("Is there a food pantry open right now near Union Square?",),
    )

    await get_tools()[0].handler({"near": "Union Square, Manhattan, NYC"}, ctx)

    assert seen == ["Union Square"]


async def test_nearest_food_pantry_accepts_city_qualifiers_added_to_one_word_neighborhood(
    monkeypatch,
):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Is there a food pantry near Flushing?",
        user_turns=("Is there a food pantry near Flushing?",),
    )

    await get_tools()[0].handler({"near": "Flushing, Queens, NYC"}, ctx)

    assert seen == ["Flushing"]


@pytest.mark.parametrize(
    ("evidence_grade", "near", "expected"),
    [
        ("authoritative", "151 East 151st Street, Bronx, NY", ["151 East 151st Street, Bronx, NY"]),
        ("discovery", "151 East 151st Street, Bronx, NY", []),
        ("authoritative", "999 Invented Street, Bronx, NY", []),
    ],
)
async def test_nearest_food_pantry_accepts_only_exact_authoritative_source_locations(
    monkeypatch,
    evidence_grade,
    near,
    expected,
):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
        snippet=(
            "Prevention Assistance and Temporary Housing (PATH) "
            "151 East 151st Street, Bronx, NY. PATH is open 24 hours."
        ),
        title="NYC DHS PATH",
        kind="WEB",
        provenance={"evidence_grade": evidence_grade},
    )
    query = "I will be at the Bronx PATH area Friday morning and need food help nearby."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    out = await get_tools()[0].handler(
        {
            "near": near,
            "near_source_citation": source_id,
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert seen == expected
    if expected:
        assert f"{{cite:{source_id}}}" in out
        assert "{{cite:" not in out
    elif evidence_grade == "authoritative":
        assert "Stop calling tools for this location" in out
        assert "exact NYC address or intersection" in out
    else:
        assert "call official_sources" in out
        assert "then retry with its new citation id" in out


async def test_nearest_food_pantry_does_not_treat_place_name_as_resident_supplied_address(
    monkeypatch,
):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("unverified expanded address reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/path",
        snippet="PATH services are available in the Bronx.",
        title="NYC PATH",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )
    query = "I will be at the Bronx PATH area Friday morning."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    out = await get_tools()[0].handler(
        {
            "near": "PATH, 151 East 151st Street, Bronx, NY",
            "near_source_citation": source_id,
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert "Stop calling tools for this location" in out
    assert "exact NYC address or intersection" in out


@pytest.mark.parametrize(
    ("evidence_grade", "expected"),
    [
        ("authoritative", ["151 East 151st Street, Bronx, NY"]),
        ("discovery", []),
    ],
)
async def test_nearest_food_pantry_recovers_source_location_without_repeated_citation_id(
    monkeypatch,
    evidence_grade,
    expected,
):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
        snippet="PATH is at 151 East 151st Street, Bronx, NY.",
        title="NYC DHS PATH",
        kind="WEB",
        provenance={"evidence_grade": evidence_grade},
    )
    query = "I will be at the Bronx PATH area Friday morning."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    out = await get_tools()[0].handler(
        {
            "near": "151 East 151st Street, Bronx, NY",
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert seen == expected
    if expected:
        assert f"{{cite:{source_id}}}" in out
    else:
        assert "call official_sources" in out
        assert f"{{cite:{source_id}}}" in out


@pytest.mark.parametrize(
    "snippet",
    [
        (
            "Prevention Assistance and Temporary Housing (PATH) "
            "151 East 151st Street Bronx, NY."
        ),
        (
            "Prevention Assistance and Temporary Housing（PATH）："
            "151 East 151st Street，Bronx，NY."
        ),
    ],
)
async def test_nearest_food_pantry_accepts_source_address_with_different_punctuation(
    monkeypatch,
    snippet,
):
    seen = []

    async def geocode_then_stop(text, **kwargs):
        seen.append(text)
        return None

    monkeypatch.setattr(fp, "geocode", geocode_then_stop)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
        snippet=snippet,
        title="NYC DHS PATH",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )
    query = "I will be at the Bronx PATH area Friday morning."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    await get_tools()[0].handler(
        {
            "near": "151 East 151st Street, Bronx, NY",
            "near_source_citation": source_id,
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert seen == ["151 East 151st Street, Bronx, NY"]


async def test_nearest_food_pantry_rejects_an_unrelated_address_from_the_same_source(
    monkeypatch,
):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("unrelated source address reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
        snippet=(
            "PATH is at 151 East 151st Street, Bronx, NY. "
            "A different office is at 200 Example Street, Bronx, NY."
        ),
        title="NYC DHS family services",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )
    query = "I will be at the Bronx PATH area Friday morning."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    out = await get_tools()[0].handler(
        {
            "near": "200 Example Street, Bronx, NY",
            "near_source_citation": source_id,
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert "Stop calling tools for this location" in out
    assert "exact NYC address or intersection" in out
    assert "200 Example Street" not in out


async def test_nearest_food_pantry_rejects_an_unrelated_address_in_the_same_sentence(
    monkeypatch,
):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("unrelated same-sentence address reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    citations = CitationRegistry()
    source_id = citations.register(
        "https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
        snippet=(
            "PATH is at 151 East 151st Street, Bronx, NY, while a different office is at "
            "200 Example Street, Bronx, NY."
        ),
        title="NYC DHS family services",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )
    query = "I will be at the Bronx PATH area Friday morning."
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=(query,),
    )

    out = await get_tools()[0].handler(
        {
            "near": "200 Example Street, Bronx, NY",
            "near_source_citation": source_id,
            "near_source_place": "PATH",
        },
        ctx,
    )

    assert "Stop calling tools for this location" in out
    assert "exact NYC address or intersection" in out
    assert "200 Example Street" not in out


async def test_nearest_food_pantry_ranks_grounds_and_links(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    now_day = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(-73.9600, 40.8000, program="Far Pantry", distadd="1 Far St",
                        distboro="Manhattan", distzip="10027", org_phone="212-555-0001",
                        type_fp="FP", program_type="FP", OBJECTID=1, GlobalID="aaaa-1"),
        _pantry_feature(-73.9910, 40.7510, program="Close Halal Pantry", distadd="2 Near Ave",
                        distboro="Manhattan", distzip="10001", org_phone="212-555-0002",
                        type_fp="FPH", program_type="FP", OBJECTID=2, GlobalID="aaaa-2",
                        **{f"fp_{now_day}_open1": "12:00 AM", f"fp_{now_day}_close1": "11:59 PM"}),
        _pantry_feature(None, None, program="No Coords", type_fp="FP", OBJECTID=3,
                        GlobalID="aaaa-3"),
    ]
    citations = CitationRegistry()
    client = _routed_client(features)
    ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Union Square", "k": 5}, ctx)
    await client.aclose()

    site_lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(site_lines) == 2                        # bad-coords row dropped
    assert "Close Halal Pantry" in site_lines[0]       # nearest first
    assert "Far Pantry" in site_lines[1]
    assert "rough estimate from the resolved place point, not a street address" in site_lines[0]
    assert "Halal" in site_lines[0]                    # dietary/access flag surfaced
    assert "open now" in site_lines[0].lower()         # open-now computed from structured hours
    assert "Immediate food need" not in out             # ordinary lookup keeps the normal ordering
    assert "Nearest City-listed food-help site candidates" in out
    assert "212-555-0002" in out                       # phone surfaced
    assert "www.google.com/maps/dir/?api=1&destination=40.75100,-73.99100" in out  # directions link
    assert "{cite:S1}" in out                          # grounded, cited
    assert citations.mapping()["S1"]["kind"] == "DATA"
    # citation is grounded in the ArcGIS source and does not fake an as-of date
    assert "arcgis" in citations.mapping()["S1"]["url"].lower()
    assert "globalid" in citations.mapping()["S1"]["url"].lower()  # row-addressed GlobalID permalink
    assert citations.mapping()["S1"]["provenance"]["record_id"] == "aaaa-2"
    assert citations.mapping()["S1"]["valid_as_of"] == ""
    assert "Source date unavailable" in out


async def test_f108_urgent_food_result_leads_with_fallback_and_lists_today_hours(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    now_day = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Nearby Pantry",
            distadd="2 Near Ave",
            distboro="Manhattan",
            distzip="10001",
            org_phone="212-555-0002",
            type_fp="FP",
            program_type="FP",
            GlobalID="urgent-near",
            EditDate="2025-11-04",
            **{
                f"fp_{now_day}_open1": "9:00 AM",
                f"fp_{now_day}_close1": "5:00 PM",
            },
        ),
        _pantry_feature(
            -73.9400,
            40.8000,
            program="Farther Open Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="urgent-far",
            **{
                f"fp_{now_day}_open1": "9:00 AM",
                f"fp_{now_day}_close1": "5:00 PM",
            },
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "k": 1,
            "urgent": True,
            "service_window_start": "12:00",
            "service_window_end": "12:01",
        },
        ctx,
    )
    await client.aclose()

    assert out.index("Immediate food need") < out.index("Nearby Pantry")
    assert "Lookup time: 2026-07-17 12:00 America/New_York {cite:S1}" in out
    assert "does not confirm food availability" in out
    assert "call the listed site before traveling" in out
    assert "call 311" in out
    assert "https://finder.nyc.gov/foodhelp" in out
    assert "Farther Open Pantry" not in out
    assert "Today's listed weekly hours: 9:00 AM-5:00 PM" in out
    assert "As of: 2025-11-04" in out
    assert (
        ctx.citations.mapping()["S1"]["provenance"]["derivation"]["temporal_basis"]
        == "weekly_schedule"
    )

    schema = get_tools()[0].parameters
    assert schema["properties"]["urgent"]["type"] == "boolean"
    assert schema["properties"]["on"]["format"] == "date"
    assert "urgent" not in schema["required"]


async def test_urgent_food_respects_the_requested_service_window(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    now_day = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Nearby Morning Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="morning-only",
            **{
                f"fp_{now_day}_open1": "9:00 AM",
                f"fp_{now_day}_close1": "2:00 PM",
            },
        ),
        _pantry_feature(
            -73.9800,
            40.7600,
            program="Evening Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="evening",
            **{
                f"fp_{now_day}_open1": "5:30 PM",
                f"fp_{now_day}_close1": "8:00 PM",
            },
        ),
        _pantry_feature(
            -73.9700,
            40.7700,
            program="Second Evening Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="evening-second",
            **{
                f"fp_{now_day}_open1": "6:00 PM",
                f"fp_{now_day}_close1": "9:00 PM",
            },
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "k": 3,
            "urgent": True,
            "service_window_start": "17:00",
            "service_window_end": "23:59",
        },
        ctx,
    )
    await client.aclose()

    assert "Evening Pantry" in out
    assert "Second Evening Pantry" not in out
    assert "Nearby Morning Pantry" not in out
    assert "requested 17:00-23:59 service window" in out
    assert "weekly schedule overlaps that window" in out
    assert "does not confirm food availability" in out
    assert "Lead with call 311 or https://finder.nyc.gov/foodhelp" in out
    assert "one nearest weekly-schedule lead, not the only nearby option" in out
    schema = get_tools()[0].parameters["properties"]
    assert schema["service_window_start"]["pattern"] == r"^\d{2}:\d{2}$"
    assert schema["service_window_end"]["pattern"] == r"^\d{2}:\d{2}$"


async def test_nearest_food_pantry_does_not_present_closed_candidates_as_open_now(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    now_day = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(
            -73.9910, 40.7510, program="Closed Pantry", distadd="2 Near Ave",
            distboro="Manhattan", distzip="10001", org_phone="212-555-0002",
            type_fp="FP", program_type="FP", OBJECTID=2, GlobalID="closed-1",
            **{f"fp_{now_day}_open1": "1:00 AM", f"fp_{now_day}_close1": "2:00 AM"},
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "urgent": True,
            "service_window_start": "12:00",
            "service_window_end": "12:01",
        },
        ctx,
    )
    await client.aclose()

    assert "no City-listed site in this feed has weekly hours establishing service" in out
    assert "call 311" in out
    assert "Do not present another time of day as an option for that window" in out
    assert "Closed Pantry" not in out
    citations = ctx.citations.mapping()
    assert "{cite:S1}" in out
    assert citations["S1"]["provenance"]["snapshot"]["scheduled_open_citywide"] == 0
    assert citations["S1"]["provenance"]["snapshot"]["scheduled_open_nearby"] == 0
    assert citations["S1"]["provenance"]["snapshot"]["origin_query"] == "Union Square"
    assert citations["S1"]["provenance"]["snapshot"]["origin_label"] == "Union Square, Manhattan"
    assert citations["S1"]["provenance"]["snapshot"]["origin_point"] == [
        40.743312,
        -73.988975,
    ]


async def test_nearest_food_pantry_distinguishes_unknown_hours_from_closed():
    features = [
        _pantry_feature(
            -73.9910, 40.7510, program="Unknown Hours Pantry", distadd="2 Near Ave",
            distboro="Manhattan", distzip="10001", type_fp="FP", program_type="FP",
            GlobalID="unknown-hours",
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "urgent": True,
            "service_window_start": "12:00",
            "service_window_end": "12:01",
        },
        ctx,
    )
    await client.aclose()

    assert "no City-listed site in this feed has weekly hours establishing service" in out
    assert "call 311" in out
    assert "Unknown Hours Pantry" not in out


async def test_nearest_food_pantry_returns_farther_open_lead_after_immediate_fallback(
    monkeypatch,
):
    monkeypatch.setattr(fp, "datetime", _Noon)
    now_day = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(
            -73.9910, 40.7510, program="Nearby Closed Pantry", distadd="2 Near Ave",
            distboro="Manhattan", distzip="10001", type_fp="FP", program_type="FP",
            GlobalID="closed-near",
            **{f"fp_{now_day}_open1": "1:00 AM", f"fp_{now_day}_close1": "2:00 AM"},
        ),
        _pantry_feature(
            -73.9400, 40.8000, program="Farther Open Pantry", distadd="9 Far Ave",
            distboro="Manhattan", distzip="10027", type_fp="FP", program_type="FP",
            GlobalID="open-far",
            **{f"fp_{now_day}_open1": "12:00 AM", f"fp_{now_day}_close1": "11:59 PM"},
        ),
        _pantry_feature(
            -73.9300, 40.8100, program="Second Farther Open Pantry", distadd="10 Far Ave",
            distboro="Manhattan", distzip="10027", type_fp="FP", program_type="FP",
            GlobalID="open-far-2",
            **{f"fp_{now_day}_open1": "12:00 AM", f"fp_{now_day}_close1": "11:59 PM"},
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "k": 1,
            "urgent": True,
            "service_window_start": "12:00",
            "service_window_end": "12:01",
        },
        ctx,
    )
    await client.aclose()

    assert "Nearby Closed Pantry" not in out
    assert "call 311" in out
    assert "finder.nyc.gov/foodhelp" in out
    assert "Farther Open Pantry" in out
    assert "Second Farther Open Pantry" not in out
    assert "call the listed site before traveling" in out
    assert "{cite:S1}" in out
    assert "{cite:S2}" in out
    assert len(ctx.citations.mapping()) == 2


async def test_nearest_food_pantry_uses_the_residents_requested_date(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    friday = _DAYS[_Noon.now().weekday()]
    saturday = _DAYS[(_Noon.now().weekday() + 1) % 7]
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Friday Only Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="friday-only",
            **{
                f"fp_{friday}_open1": "9:00 AM",
                f"fp_{friday}_close1": "5:00 PM",
            },
        ),
        _pantry_feature(
            -73.9800,
            40.7600,
            program="Saturday Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="saturday",
            **{
                f"fp_{saturday}_open1": "10:00 AM",
                f"fp_{saturday}_close1": "2:00 PM",
            },
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "k": 1, "on": "2026-07-18"},
        ctx,
    )
    await client.aclose()

    assert "Saturday Pantry" in out
    assert "Friday Only Pantry" not in out
    assert "Saturday, 2026-07-18" in out
    assert "scheduled open now" not in out
    assert "10:00 AM-2:00 PM" in out


async def test_f185_future_date_respects_monthly_occurrence_notes(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="First And Third Thursday Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="monthly-thursday",
            fp_notes="1ST & 3RD THURSDAY",
            fp_thu_open1="12:00 PM",
            fp_thu_close1="3:00 PM",
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "k": 1, "on": "2026-08-13"},
        ctx,
    )
    await client.aclose()

    assert "monthly schedule excludes Thursday, 2026-08-13" in out
    assert "weekly schedule lists hours on Thursday, 2026-08-13" not in out
    assert "Thursday's listed weekly hours" not in out


def test_f185_calendar_guard_rejects_wrong_monthly_weekday_occurrence():
    result = check_grounding(
        "August 13 is the third Thursday. {cite:S1}",
        {"S1": {"snippet": "1ST & 3RD THURSDAY"}},
        current_date=date(2026, 8, 9),
    )

    assert result is not None and result.blocking
    assert result.hard_failures[0].kind == "calendar_consistency"


async def test_nearest_food_pantry_filters_source_service_type():
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Nearby Soup Kitchen",
            type_sk="SK",
            program_type="SK",
            GlobalID="soup-kitchen",
        ),
        _pantry_feature(
            -73.9800,
            40.7600,
            program="Food Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="food-pantry",
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "service_type": "pantry"},
        ctx,
    )
    await client.aclose()

    assert "Food Pantry" in out
    assert "Nearby Soup Kitchen" not in out
    assert get_tools()[0].parameters["properties"]["service_type"]["enum"] == [
        "pantry",
        "soup_kitchen",
        "any",
    ]


async def test_f199_follow_up_keeps_the_previously_cited_pantry():
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Closer Different Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="closer-site",
        ),
        _pantry_feature(
            -73.9800,
            40.7600,
            program="Previously Cited Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="prior-site",
            EditDate="2026-08-10",
            fp_tue_open1="10:00 AM",
            fp_tue_close1="06:00 PM",
        ),
    ]
    citations = CitationRegistry()
    prior_id = citations.register(
        f"{fp.FOODHELP_URL}/query?where=GlobalID%3D%27prior-site%27",
        snippet="Previously Cited Pantry, 9 Example Street",
        title="NYC FoodHelp (Food Help Programs)",
        kind="DATA",
        provenance={"record_id": "prior-site", "snapshot": {}},
    )
    client = _routed_client(features)
    query = "What hours does the place you just gave me have today?"
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        http=client,
        query=query,
        user_turns=("I am near Union Square.", query),
    )

    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "site_citation": prior_id,
            "urgent": True,
            "service_window_start": "17:00",
            "service_window_end": "23:59",
        },
        ctx,
    )
    await client.aclose()

    assert "Previously Cited Pantry" in out
    assert "Closer Different Pantry" not in out
    availability = next(
        citation
        for citation in citations.mapping().values()
        if citation["title"] == "NYC FoodHelp availability lookup"
    )
    snapshot = availability["provenance"]["snapshot"]
    assert snapshot["referenced_site"]["GlobalID"] == "prior-site"
    assert snapshot["referenced_site_valid_as_of"] == "2026-08-10"


async def test_f199_rejects_a_non_foodhelp_site_reference(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("invalid site reference reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    citations = CitationRegistry()
    unrelated_id = citations.register(
        "https://example.org/not-foodhelp",
        snippet="Unrelated place",
        title="Unrelated source",
        kind="DATA",
        provenance={"record_id": "prior-site", "snapshot": {}},
    )
    query = "What hours does the place you just gave me have today?"
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query=query,
        user_turns=("I am near Union Square.", query),
    )

    out = await get_tools()[0].handler(
        {"near": "Union Square", "site_citation": unrelated_id},
        ctx,
    )

    assert "does not identify a prior NYC FoodHelp site" in out


async def test_nearest_food_pantry_labels_soup_kitchen_results_as_soup_kitchens():
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Nearby Soup Kitchen",
            type_sk="SK",
            program_type="SK",
            GlobalID="soup-kitchen",
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "service_type": "soup_kitchen"},
        ctx,
    )
    await client.aclose()

    assert "Nearest City-listed soup kitchen candidates" in out
    assert "food pantry candidates" not in out


async def test_future_same_weekday_is_not_labeled_today(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    friday = _DAYS[_Noon.now().weekday()]
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Friday Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="friday-pantry",
            **{
                f"fp_{friday}_open1": "9:00 AM",
                f"fp_{friday}_close1": "5:00 PM",
            },
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "service_type": "pantry", "on": "2026-07-24"},
        ctx,
    )
    await client.aclose()

    assert "Friday's listed weekly hours" in out
    assert "Today's listed weekly hours" not in out


async def test_nearest_food_pantry_excludes_unknown_source_types_from_typed_request():
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Unknown Program",
            type_fp="FP",
            program_type="",
            GlobalID="unknown",
        ),
        _pantry_feature(
            -73.9900,
            40.7520,
            program="Malformed Program",
            type_fp="FP",
            program_type="OTHER",
            GlobalID="malformed",
        ),
        _pantry_feature(
            -73.9800,
            40.7600,
            program="Verified Food Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="food-pantry",
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "service_type": "pantry"},
        ctx,
    )
    await client.aclose()

    assert "Verified Food Pantry" in out
    assert "Unknown Program" not in out
    assert "Malformed Program" not in out


async def test_nearest_food_pantry_rejects_past_service_date_before_lookup(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)

    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("past date reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await get_tools()[0].handler(
        {"near": "Union Square", "on": "2026-07-16"},
        ctx,
    )

    assert "cannot verify a past service date" in out


async def test_nearest_food_pantry_flags_conflicting_schedule_fields(monkeypatch):
    monkeypatch.setattr(fp, "datetime", _Noon)
    saturday = _DAYS[(_Noon.now().weekday() + 1) % 7]
    features = [
        _pantry_feature(
            -73.9910,
            40.7510,
            program="Conflicting Pantry",
            type_fp="FP",
            program_type="FP",
            GlobalID="conflicting",
            fp_days_orig="SAT(2ND,4TH)",
            fp_notes="ONLY OPEN 1ST & 3RD SATURDAY",
            **{
                f"fp_{saturday}_open1": "10:00 AM",
                f"fp_{saturday}_close1": "2:00 PM",
            },
        ),
    ]
    client = _routed_client(features)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {"near": "Union Square", "k": 1, "on": "2026-07-18"},
        ctx,
    )
    await client.aclose()

    assert "source fields conflict about the Saturday schedule" in out
    assert "does not confirm service that day" in out


async def test_nearest_food_pantry_cites_an_empty_official_feed():
    client = _routed_client([])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)

    out = await get_tools()[0].handler(
        {
            "near": "Union Square",
            "urgent": True,
            "service_window_start": "12:00",
            "service_window_end": "12:01",
        },
        ctx,
    )
    await client.aclose()

    assert "No open food-help sites came back" in out
    assert "{cite:S1}" in out
    citation = ctx.citations.mapping()["S1"]
    assert ctx.response_priority_citation_ids == {"S1"}
    assert citation["valid_as_of"] == ""
    assert citation["provenance"]["snapshot"]["citywide_records_checked"] == 0


async def test_nearest_food_pantry_abstains_when_geocode_fails(monkeypatch):
    async def fail(text, **kwargs):
        return None
    monkeypatch.setattr(fp, "geocode", fail)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": []})))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Springfield, Illinois"}, ctx)
    await client.aclose()

    assert "- " not in out                              # no fabricated pantry list
    low = out.lower()
    assert "couldn't" in low or "could not" in low
    assert "nyc" in low


async def test_nearest_food_pantry_clarifies_on_low_confidence(monkeypatch):
    async def ambiguous(text, **kwargs):
        return GeoPoint(40.7, -73.9, "ambiguous", low_confidence=True)
    monkeypatch.setattr(fp, "geocode", ambiguous)

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]), http=client)
    out = await get_tools()[0].handler({"near": "Broadway and 100th"}, ctx)
    await client.aclose()
    assert "which borough" in out.lower()
    assert "- " not in out


# --- the shipped module stays valid ---------------------------------------

def test_food_pantries_module_loads_with_tool_and_eval():
    registry = Registry.discover(config.MODULES_DIR)
    module = next((m for m in registry.modules if m.name == "food_pantries"), None)
    assert module is not None
    assert module.category == "Food"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "nearest_food_pantry" in tool_names

    from heynyc.eval.cases import load_cases
    cases = [c for c in load_cases(registry) if c.module == "food_pantries"]
    assert cases, "food_pantries should ship eval cases"
    provenance_case = next(c for c in cases if c.id == "food_source_origin_clarifies")
    assert provenance_case.invariants == {"allow_clarification": True}
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)


# F159: assistant proposed a location, resident confirmed, assistant re-asked
# Not a model or memory failure; the clarification IS in history
# The origin guard was handed `user_turns` and passed `()`
def test_origin_may_come_from_a_location_the_resident_gave_earlier():
    turns = (
        "where is the nearest food pantry to 82nd St and Roosevelt Ave in Queens?",
        "yes I am right there",
    )

    assert fp._resident_supplied_origin(
        "82nd St and Roosevelt Ave", "yes I am right there", turns
    ) == "82nd St and Roosevelt Ave"


def test_origin_the_resident_never_gave_is_still_rejected():
    """The anti-hallucination property must survive: history widens WHERE we look, not WHAT counts."""
    turns = (
        "where is the nearest food pantry to 82nd St and Roosevelt Ave in Queens?",
        "yes I am right there",
    )

    assert fp._resident_supplied_origin("350 Fifth Avenue", "yes I am right there", turns) == ""


def test_a_newly_named_location_is_preferred_over_the_earlier_one():
    """Inverse: the resident moving on must not be overridden by a stale prior location."""
    turns = (
        "where is the nearest food pantry to 82nd St and Roosevelt Ave in Queens?",
        "what about a food pantry in the Bronx?",
    )

    assert fp._resident_supplied_origin(
        "the Bronx", "what about a food pantry in the Bronx?", turns
    ) == "the Bronx"


def test_f195_multilingual_intersection_survives_identity_preserving_model_wording():
    query = (
        "Ya me mudé cerca de la calle 82 y Roosevelt Avenue en Queens. "
        "¿Qué opciones tengo aquí?"
    )

    assert fp._resident_supplied_origin(
        "near 82nd Street and Roosevelt Avenue, Queens, NY",
        query,
        (query,),
    ) == "near 82nd Street and Roosevelt Avenue, Queens, NY"


def test_f195_multilingual_intersection_rejects_a_model_changed_borough():
    query = "Estoy en la calle 82 y Roosevelt Avenue en Queens."

    assert fp._resident_supplied_origin(
        "82nd Street and Roosevelt Avenue, Brooklyn, NY",
        query,
        (query,),
    ) == ""
