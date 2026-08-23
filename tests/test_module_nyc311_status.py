"""311 read-only service-request status lane.

Two grounded intents against the keyless Socrata dataset erm2-nwe9:
  1. "is my complaint moving?" -> lookup by the resident's SR number (unique_key),
  2. "what's happening with complaints about X near me" -> topic + area movement.
Read-only: filing (Create-SR) is out of scope. Fully offline: query_dataset and
geocode are injected, never a live Socrata/geocoder call.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

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


def test_area_search_uses_typed_terms_and_project_wide_result_count_name():
    tool = nyc311.get_tools()[1]

    from heynyc.core.location import LocationRequest

    assert issubclass(nyc311.ComplaintSearchQuery, LocationRequest)

    assert set(tool.parameters["properties"]) == {
        "complaint_terms",
        "near",
        "max_results",
        "within_days",
        "radius_meters",
    }
    assert tool.parameters["required"] == ["complaint_terms"]
    terms = tool.parameters["properties"]["complaint_terms"]
    assert terms["type"] == "array"
    assert terms["minItems"] == 1
    assert terms["maxItems"] == 3
    assert terms["items"]["minLength"] == 4
    assert "dataset-facing" in terms["description"]
    assert "rodent" in terms["description"]
    assert tool.parameters["properties"]["within_days"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 365,
        "default": 30,
        "description": "Lookback window in elapsed days; omit to use 30 days",
        "title": "Within Days",
    }
    assert tool.parameters["properties"]["radius_meters"] == {
        "type": "integer",
        "minimum": 100,
        "maximum": 50_000,
        "default": 800,
        "description": "Search radius around near, in meters; omit to use 800 meters",
        "title": "Radius Meters",
    }


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
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", boom_geocode)

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
    monkeypatch.setattr("heynyc.core.tools.geo.geocode", boom_geocode)

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
        if kwargs.get("group") == "status":
            return [
                {"status": "In Progress", "count": "1"},
                {"status": "Closed", "count": "1"},
            ]
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

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)
    monkeypatch.setattr(
        nyc311,
        "_nyc_now",
        lambda: datetime(2026, 8, 19, 15, 30, tzinfo=ZoneInfo("America/New_York")),
    )

    ctx = _ctx()
    out = await nyc311.get_tools()[1].handler(
        {
            "complaint_terms": ["noise", "loud music"],
            "near": "Union Square",
            "max_results": 5,
        },
        ctx,
    )

    where = captured[0]["where"]
    assert captured[0]["select"] == "status, count(*) as count"
    assert captured[0]["group"] == "status"
    assert captured[0]["limit"] is None
    assert captured[0]["exclude_system_fields"] is None
    assert captured[1]["limit"] == 5
    assert captured[1]["order"] == "created_date DESC"
    assert "within_circle(location, 40.7359, -73.9911," in where
    assert "upper(complaint_type) like upper('%noise%')" in where
    assert "upper(descriptor) like upper('%loud music%')" in where
    assert " OR " in where
    assert "created_date > '2026-07-20T15:30:00'" in where
    assert "800 meters (about 0.5 mile)" in out
    assert "In Progress" in out
    assert "Closed" in out
    assert "{cite:S1}" in out
    assert "{cite:S2}" in out
    assert "{cite:S1} {cite:S2}" not in out
    assert "opened 2026-07-17 {cite:S2}" in out
    assert "once a day" not in out.lower()
    assert len(ctx.citations.mapping()) == 3
    aggregate = ctx.citations.mapping()["S1"]
    assert aggregate["provenance"]["record_id"] == "area-search"
    assert aggregate["provenance"]["field_pointer"] == "/"
    assert aggregate["provenance"]["snapshot"]["status_counts"] == {
        "In Progress": 1,
        "Closed": 1,
    }
    assert aggregate["provenance"]["snapshot"]["examples"]
    assert aggregate["valid_as_of"] == "2026-08-19T15:30:00-04:00"
    assert "status:" in aggregate["snippet"]
    assert "2 total matches; 2 most recent examples shown" in aggregate["snippet"]
    assert "800 meters (about 0.5 mile)" in aggregate["snippet"]
    assert "last 30 days" in aggregate["snippet"]
    assert aggregate["provenance"]["derivation"] == {
        "where": where,
        "aggregate": {
            "select": "status, count(*) as count",
            "group": "status",
        },
        "origin": {
            "label": "Union Square, Manhattan",
            "latitude": 40.7359,
            "longitude": -73.9911,
        },
        "checked_at": "2026-08-19T15:30:00-04:00",
        "examples": {
            "order": "created_date DESC",
            "limit": 5,
            "exclude_system_fields": False,
        },
    }
    assert "%24select=status%2C+count%28%2A%29+as+count" in aggregate["url"]
    assert "%24group=status" in aggregate["url"]
    assert "200" not in out


@pytest.mark.asyncio
async def test_area_lookup_honors_explicit_time_window_and_radius(monkeypatch):
    captured: list[dict] = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square, Manhattan")

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)
    monkeypatch.setattr(
        nyc311,
        "_nyc_now",
        lambda: datetime(2026, 8, 19, 15, 30, tzinfo=ZoneInfo("America/New_York")),
    )

    out = await nyc311.get_tools()[1].handler(
        {
            "complaint_terms": ["noise"],
            "near": "Union Square",
            "within_days": 7,
            "radius_meters": 1600,
        },
        _ctx(),
    )

    assert "created_date > '2026-08-12T15:30:00'" in captured[0]["where"]
    assert "within_circle(location, 40.7359, -73.9911, 1600)" in captured[0]["where"]
    assert "last 7 days" in out
    assert "within 1600 meters (about 1 mile)" in out


@pytest.mark.parametrize(
    ("now", "expected_cutoff"),
    [
        (datetime(2026, 3, 10, 12, 0, tzinfo=ZoneInfo("America/New_York")), "2026-03-03T11:00:00"),
        (datetime(2026, 11, 3, 12, 0, tzinfo=ZoneInfo("America/New_York")), "2026-10-27T13:00:00"),
    ],
)
@pytest.mark.asyncio
async def test_area_lookup_uses_elapsed_days_across_dst(
    monkeypatch,
    now: datetime,
    expected_cutoff: str,
):
    captured: list[dict] = []

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)
    monkeypatch.setattr(nyc311, "_nyc_now", lambda: now)

    await nyc311.get_tools()[1].handler(
        {"complaint_terms": ["noise"], "within_days": 7},
        _ctx(),
    )

    assert f"created_date > '{expected_cutoff}'" in captured[0]["where"]


@pytest.mark.asyncio
async def test_area_lookup_escapes_topic_quotes(monkeypatch):
    captured: list[dict] = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square")

    async def fake_qd(dataset_id, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    await nyc311.get_tools()[1].handler(
        {"complaint_terms": ["o'brien's"], "near": "here"}, _ctx()
    )

    # SoQL string literals escape a quote by doubling it, so the topic can't break the query.
    assert "upper('%o''brien''s%')" in captured[0]["where"]


@pytest.mark.asyncio
async def test_area_lookup_rejects_short_fragments_before_querying(monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("invalid complaint terms must not reach a data service")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", boom)
    monkeypatch.setattr(nyc311, "query_dataset", boom)

    out = await nyc311.get_tools()[1].handler(
        {"complaint_terms": ["rat"], "near": "Union Square"}, _ctx()
    )

    assert "at least 4 characters" in out


@pytest.mark.asyncio
async def test_area_lookup_low_confidence_location_asks_to_clarify(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, "somewhere", low_confidence=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)

    out = await nyc311.get_tools()[1].handler(
        {"complaint_terms": ["noise"], "near": "the park"}, _ctx()
    )

    assert "specific NYC address" in out


@pytest.mark.asyncio
async def test_area_lookup_no_matches_is_honest_and_routes_to_311(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7359, -73.9911, "Union Square")

    async def fake_qd(dataset_id, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(nyc311, "query_dataset", fake_qd)

    ctx = _ctx()
    out = await nyc311.get_tools()[1].handler(
        {"complaint_terms": ["noise"], "near": "Union Square"}, ctx
    )

    assert "no" in out.lower()
    assert "311" in out
    assert "{cite:S1}" in out
    citation = ctx.citations.mapping()["S1"]
    assert citation["kind"] == "DATA"
    assert citation["provenance"]["field_pointer"] == "/"
    assert citation["provenance"]["snapshot"] == {
        "status_counts": {},
        "examples": [],
    }
    assert "within_circle" in citation["url"]


@pytest.mark.asyncio
async def test_needs_an_sr_number_or_topic_or_location():
    out = await nyc311.get_tools()[1].handler({}, _ctx())

    assert "service request number" in out.lower() or "SR number" in out
