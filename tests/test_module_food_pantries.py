"""Offline tests for the food_pantries module.

Grounded in the city's FoodHelp ArcGIS backend, but every HTTP call is mocked/injected —
no live ArcGIS or geocoder call. Covers: ranking by distance, open-now computation from the
structured fp_<day>_open*/close* hours, dietary/access flags, the directions link, a grounded
DATA citation, and abstention when geocoding fails.
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
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


def test_open_now_handles_cross_midnight_hours():
    late = datetime(2026, 7, 1, 23, 30)
    late_day = _DAYS[late.weekday()]
    overnight = _hours_record(late_day, "10:00 PM", "2:00 AM")
    assert _open_now(overnight, late) is True

    early = datetime(2026, 7, 2, 1, 30)
    assert _open_now(overnight, early) is True
    assert _open_now(overnight, datetime(2026, 7, 2, 3, 0)) is False


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


async def test_nearest_food_pantry_rejects_location_from_prior_turn(monkeypatch):
    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("stale location reached geocoder")

    monkeypatch.setattr(fp, "geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="Which one is open now?",
        user_history="I am near Jackson Heights.\nWhich one is open now?",
        user_turns=("I am near Jackson Heights.", "Which one is open now?"),
    )

    out = await get_tools()[0].handler({"near": "Jackson Heights"}, ctx)

    assert "proposed search origin was not supplied" in out


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
    assert "Halal" in site_lines[0]                    # dietary/access flag surfaced
    assert "open now" in site_lines[0].lower()         # open-now computed from structured hours
    assert "Immediate food need" not in out             # ordinary lookup keeps the normal ordering
    assert "Nearest City-listed food pantry candidates" in out
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
        {"near": "Union Square", "k": 1, "urgent": True},
        ctx,
    )
    await client.aclose()

    assert out.index("Immediate food need") < out.index("Nearby Pantry")
    assert "does not confirm food availability now or later today" in out
    assert "call the listed site now" in out
    assert "call 311" in out
    assert "https://finder.nyc.gov/foodhelp" in out
    assert "offer to search farther" in out
    assert "Farther Open Pantry" not in out
    assert "Today's listed weekly hours: 9:00 AM-5:00 PM" in out
    assert "As of: 2025-11-04" in out
    assert (
        ctx.citations.mapping()["S1"]["provenance"]["derivation"]["temporal_basis"]
        == "weekly_schedule"
    )

    schema = get_tools()[0].parameters
    assert schema["properties"]["urgent"]["type"] == "boolean"
    assert "urgent" not in schema["required"]


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
        {"near": "Union Square", "urgent": True},
        ctx,
    )
    await client.aclose()

    assert "no City-listed site in this feed is scheduled open now" in out
    assert "call 311" in out
    assert "Do not show closed-site cards or offer to search farther in the same feed" in out
    assert "Closed Pantry" not in out
    assert not ctx.citations.mapping()


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
        {"near": "Union Square", "urgent": True},
        ctx,
    )
    await client.aclose()

    assert "hours are unavailable" in out.lower()
    assert "may still be open" in out
    assert "No City-listed site in this feed is scheduled open now" not in out
    assert "call-only leads" in out
    assert "not travel unless a site confirms service" in out


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
        {"near": "Union Square", "k": 1, "urgent": True},
        ctx,
    )
    await client.aclose()

    assert "Nearby Closed Pantry" not in out
    assert "call 311" in out
    assert "finder.nyc.gov/foodhelp" in out
    assert "Farther Open Pantry" in out
    assert "Second Farther Open Pantry" not in out
    assert "farther scheduled-open lead" in out
    assert "call before traveling" in out
    assert "{cite:S1}" in out
    assert len(ctx.citations.mapping()) == 1


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
    assert any(c.invariants.get("must_abstain_or_redirect") for c in cases)
