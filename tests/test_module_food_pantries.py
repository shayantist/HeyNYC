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


async def test_nearest_food_pantry_ranks_grounds_and_links():
    now_day = _DAYS[datetime.now().weekday()]
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
