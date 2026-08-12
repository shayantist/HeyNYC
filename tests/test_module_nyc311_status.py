"""311 read-only service-request status lane.

Two grounded intents against the keyless Socrata dataset erm2-nwe9:
  1. "is my complaint moving?" -> lookup by the resident's SR number (unique_key),
  2. "what's happening with complaints about X near me" -> topic + area movement.
Read-only: filing (Create-SR) is out of scope. Fully offline: query_dataset and
geocode are injected, never a live Socrata/geocoder call.
"""
from __future__ import annotations

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.nyc311_status import tools as nyc311


def _ctx() -> ToolContext:
    return ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )


def _closed_sr() -> dict:
    return {
        ":id": "row-7xfz-fckg.gxme",
        ":updated_at": "2026-07-18T01:53:30.950Z",
        "unique_key": "69741503",
        "created_date": "2026-07-17T01:57:48.000",
        "closed_date": "2026-07-17T02:04:15.000",
        "complaint_type": "Noise - Residential",
        "descriptor": "Loud Music/Party",
        "status": "Closed",
        "resolution_description": (
            "The Police Department responded and determined that no violation existed."
        ),
        "resolution_action_updated_date": "2026-07-17T02:04:22.000",
        "agency_name": "New York City Police Department",
        "borough": "BROOKLYN",
    }


def test_module_loads_custom_tool():
    names = {tool.name for tool in Registry.discover(config.MODULES_DIR).load_module_tools()}
    assert {"check_311_request", "search_311_complaints"} <= names


def test_module_declares_the_311_dataset_binding():
    binding = Registry.discover(config.MODULES_DIR).dataset_bindings().get("service_request_311")
    assert binding is not None
    assert binding.id == "erm2-nwe9"


def test_sr_permalink_is_refetchable_unique_key_query():
    url = nyc311._sr_permalink("69741503")
    assert url.startswith("https://data.cityofnewyork.us/resource/erm2-nwe9.json")
    assert "unique_key='69741503'" in url.replace("%27", "'").replace("%3D", "=")


def test_valid_as_of_uses_records_own_movement_date():
    assert nyc311._valid_as_of(_closed_sr()) == "2026-07-17T02:04:22.000"
    assert (
        nyc311._valid_as_of({"created_date": "2026-07-01T00:00:00.000"})
        == "2026-07-01T00:00:00.000"
    )
    assert (
        nyc311._valid_as_of(
            {"created_date": "2026-07-01T00:00:00.000", "closed_date": "2026-07-05T00:00:00.000"}
        )
        == "2026-07-05T00:00:00.000"
    )


@pytest.mark.asyncio
async def test_sr_lookup_reports_status_resolution_and_cites_full_row(monkeypatch):
    captured: list[dict] = []

    async def fake_qd(dataset_id, **kwargs):
        captured.append({"dataset_id": dataset_id, **kwargs})
        return [_closed_sr()]

    async def boom_geocode(*args, **kwargs):
        raise AssertionError("an SR-number lookup must never geocode a resident's address")

    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)
    monkeypatch.setattr(nyc311, "geocode", boom_geocode)

    ctx = _ctx()
    out = await nyc311.get_tools()[0].handler({"sr_number": "69741503"}, ctx)

    assert captured[0]["dataset_id"] == "erm2-nwe9"
    assert captured[0]["where"] == "unique_key='69741503'"
    assert "69741503" in out
    assert "Closed" in out
    assert "no violation existed" in out
    assert "{cite:S1}" in out
    assert "refile" in out.lower()
    assert "feedback" in out.lower()
    assert "{cite:S2}" in out
    assert "once a day" not in out.lower()

    cites = ctx.citations.mapping()
    assert len(cites) == 2
    cite = cites["S1"]
    assert cite["kind"] == "DATA"
    assert cite["valid_as_of"] == "2026-07-17T02:04:22.000"
    assert cite["provenance"]["record_id"] == "69741503"
    # full-row snapshot, not a hand-picked subset
    assert cite["provenance"]["snapshot"]["complaint_type"] == "Noise - Residential"
    assert cite["provenance"]["snapshot"]["status"] == "Closed"
    assert cites["S2"]["url"] == "https://portal.311.nyc.gov/article/?kanumber=KA-02419"
    assert cites["S2"]["kind"] == "DOC"


@pytest.mark.asyncio
async def test_sr_lookup_sends_only_the_digits_never_stray_text(monkeypatch):
    captured: list[dict] = []

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    out = await nyc311.get_tools()[0].handler({"sr_number": "SR# 6974-1503 !!"}, _ctx())

    # PII/injection boundary: only the SR digits reach Socrata, nothing else.
    assert captured[0]["where"] == "unique_key='69741503'"
    assert "311" in out  # not found -> route to 311, never fabricate


@pytest.mark.asyncio
async def test_status_tool_never_geocodes(monkeypatch):
    captured: list[dict] = []

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    async def boom_geocode(*args, **kwargs):
        raise AssertionError("SR number present: must not geocode")

    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)
    monkeypatch.setattr(nyc311, "geocode", boom_geocode)

    await nyc311.get_tools()[0].handler(
        {"sr_number": "69741503"}, _ctx()
    )

    assert captured[0]["where"] == "unique_key='69741503'"


@pytest.mark.asyncio
async def test_sr_not_found_routes_to_311_without_fabricating(monkeypatch):
    async def fake_qd(dataset_id, **kwargs):
        return []

    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    out = await nyc311.get_tools()[0].handler({"sr_number": "12345678"}, _ctx())

    assert "12345678" in out
    assert "311" in out
    assert "Closed" not in out and "In Progress" not in out  # nothing invented


@pytest.mark.asyncio
async def test_area_lookup_filters_by_type_and_geo_and_cites_each(monkeypatch):
    captured: list[dict] = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square, Manhattan")

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return [
            {
                ":id": "r1", "unique_key": "1", "created_date": "2026-07-17T00:16:55.000",
                "complaint_type": "Noise - Residential", "descriptor": "Loud Music/Party",
                "status": "In Progress", "latitude": "40.7392", "longitude": "-73.9841",
            },
            {
                ":id": "r2", "unique_key": "2", "created_date": "2026-07-16T23:13:44.000",
                "complaint_type": "Noise - Commercial", "descriptor": "Loud Music/Party",
                "status": "Closed", "resolution_description": "Summons issued.",
                "resolution_action_updated_date": "2026-07-16T23:59:00.000",
                "latitude": "40.7333", "longitude": "-73.9898",
            },
        ]

    monkeypatch.setattr(nyc311, "geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    ctx = _ctx()
    out = await nyc311.get_tools()[1].handler({"about": "noise", "near": "Union Square"}, ctx)

    where = captured[0]["where"]
    assert "within_circle(location, 40.7359, -73.9911," in where
    assert "upper(complaint_type) like upper('%noise%')" in where
    assert "created_date >" in where
    assert "In Progress" in out
    assert "Closed" in out
    assert "{cite:S1}" in out
    assert "{cite:S2}" in out
    assert "once a day" not in out.lower()
    assert len(ctx.citations.mapping()) == 2


@pytest.mark.asyncio
async def test_area_lookup_escapes_topic_quotes(monkeypatch):
    captured: list[dict] = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square")

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(nyc311, "geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    await nyc311.get_tools()[1].handler({"about": "o'brien's", "near": "here"}, _ctx())

    # SoQL string literals escape a quote by doubling it, so the topic can't break the query.
    assert "upper('%o''brien''s%')" in captured[0]["where"]


@pytest.mark.asyncio
async def test_area_lookup_low_confidence_location_asks_to_clarify(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, "somewhere", low_confidence=True)

    monkeypatch.setattr(nyc311, "geocode", fake_geocode)

    out = await nyc311.get_tools()[1].handler({"about": "noise", "near": "the park"}, _ctx())

    assert "specific NYC address" in out


@pytest.mark.asyncio
async def test_area_lookup_no_matches_is_honest_and_routes_to_311(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square")

    async def fake_qd(dataset_id, **kwargs):
        return []

    monkeypatch.setattr(nyc311, "geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    out = await nyc311.get_tools()[1].handler({"about": "noise", "near": "Union Square"}, _ctx())

    assert "no" in out.lower()
    assert "311" in out


@pytest.mark.asyncio
async def test_needs_an_sr_number_or_topic_or_location():
    out = await nyc311.get_tools()[1].handler({}, _ctx())

    assert "service request number" in out.lower() or "SR number" in out
