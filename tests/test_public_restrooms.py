"""Public-restroom lookup chooses useful, corroborated options."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext, ToolFailure
from heynyc.core.tools.geo import GeoPoint
from heynyc.core.tools.geocoder import GeocoderUnavailable
from heynyc.modules.public_restrooms import tools as restrooms


def test_public_restrooms_module_loads_custom_lookup():
    registry = Registry.discover(config.MODULES_DIR)

    tool_names = {tool.name for tool in registry.load_module_tools()}

    assert "find_public_restrooms" in tool_names


def test_public_restrooms_scope_accessibility_and_311_claims():
    registry = Registry.discover(config.MODULES_DIR)
    module = next(item for item in registry.modules if item.name == "public_restrooms")
    properties = restrooms.get_tools()[0]._input_schema()["properties"]
    parameter = properties["fully_accessible"]
    prompt = " ".join(module.prompt.split())

    assert "site accessibility field" in parameter["description"]
    assert "does not prove restroom fixture accessibility" in parameter["description"]
    assert "open_now_only" not in properties
    assert "schedule evidence is context" in prompt
    assert "Only route an explicitly reported closure or maintenance problem" in prompt
    assert "include every closer partial match" in prompt
    assert "call web_fetch once" in prompt
    assert "Never call a result a confirmed fully accessible restroom" in prompt
    assert "Repeat that fixture-accessibility limit beside every result" in prompt


@pytest.mark.asyncio
async def test_lookup_keeps_distance_order_when_farther_schedule_is_known(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "near",
                ":updated_at": "2025-11-05T12:00:00.000Z",
                "facility_name": "Unverified Lobby",
                "latitude": "40.7581",
                "longitude": "-73.9780",
                "status": "Operational",
                "hours_of_operation": "Weekdays 9am-5pm",
            },
            {
                ":id": "open",
                ":updated_at": "2025-11-05T12:00:00.000Z",
                "facility_name": "645 Fifth Avenue POPS",
                "latitude": "40.7592",
                "longitude": "-73.9761",
                "status": "Operational",
                "hours_of_operation": "Everyday 8am-10pm",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        return [
            {
                "OBJECTID": 17439,
                "NYCEM_ID": "CO016",
                "Facility_name": "POPS - 645 Fifth Avenue",
                "lat": 40.7592,
                "lon": -73.9761,
                "Finder_status": "OPEN",
                "Location_type": "Indoor",
                "Space_type": "Other Indoor Cool Option",
                "Accessible": "Yes",
                "Entrance_information": "Privately Owned Public Space. Hours are subject to change.",
                "Wednesday": "8a-10p",
                "cc_wed_open1": "08:00 AM",
                "cc_wed_close1": "10:00 PM",
            }
        ]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 13, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    registry = Registry.discover(config.MODULES_DIR)
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    output = await restrooms.get_tools()[0].handler(
        {"near": "Rockefeller Center", "max_results": 1}, ctx
    )

    assert "1. Unverified Lobby" in output
    assert "645 Fifth Avenue POPS" not in output
    assert "not independently confirmed open now" in output
    assert "NYC restroom record date: 2025-11-05" in output
    assert "{cite:S1}" in output
    assert len(ctx.citations.mapping()) == 1
    assert all(
        citation["provenance"]["derivation"]["origin_label"] == "Rockefeller Center"
        for citation in ctx.citations.mapping().values()
    )


@pytest.mark.asyncio
async def test_nearest_restroom_does_not_let_farther_schedule_evidence_beat_distance(
    monkeypatch,
):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.57328, -73.97033, text)

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "coney",
                "facility_name": "Coney Island Beach Zone 4b",
                "latitude": "40.57341",
                "longitude": "-73.97590",
                "status": "Operational",
                "hours_of_operation": "8am-4pm, open later seasonally",
            },
            {
                ":id": "maiden",
                "facility_name": "180 Maiden Lane POPS",
                "latitude": "40.70528",
                "longitude": "-74.00552",
                "status": "Operational",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        return [{
            "OBJECTID": 17445,
            "Facility_name": "180 Maiden Lane POPS",
            "lat": 40.70525,
            "lon": -74.00547,
            "Finder_status": "OPEN",
            "cc_fri_open1": "08:00 AM",
            "cc_fri_close1": "10:00 PM",
        }]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 8, 21, 17, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler(
        {"near": "Coney Island Beach", "max_results": 1}, ctx
    )

    assert "1. Coney Island Beach Zone 4b" in output
    assert "180 Maiden Lane POPS" not in output


@pytest.mark.asyncio
async def test_lookup_keeps_unverified_nearby_options_and_caps_requested_limit(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": str(index),
                "facility_name": f"Restroom {index}",
                "latitude": str(40.0 + index / 10_000),
                "longitude": "-73.0",
                "status": "Operational",
            }
            for index in range(12)
        ]

    async def fake_cool_options(*args, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    registry = Registry.discover(config.MODULES_DIR)
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    output = await restrooms.get_tools()[0].handler({"near": "here", "max_results": 10}, ctx)

    assert "10. Restroom 9" in output
    assert "Restroom 10" not in output
    assert "not independently confirmed open now" in output


@pytest.mark.asyncio
async def test_lookup_surfaces_city_access_and_family_details_without_corroboration(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [{
            ":id": "1",
            "facility_name": "City Restroom",
            "latitude": "40.001",
            "longitude": "-73.0",
            "status": "Operational",
            "open": "Seasonal",
            "hours_of_operation": "8am-4pm, open later seasonally",
            "accessibility": "Fully Accessible",
            "restroom_type": "Single-Stall All Gender Restroom(s)",
            "changing_stations": "Yes",
            "additional_notes": "A key is needed to enter",
            "website": {"url": "https://example.nyc/restroom"},
        }]

    async def fake_cool_options(*args, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler({"near": "here"}, ctx)

    assert "Seasonal" in output
    assert "NYC listing accessibility: Fully Accessible" in output
    assert "Restroom type: Single-Stall All Gender Restroom(s)" in output
    assert "Changing station: Yes" in output
    assert "Access note: A key is needed to enter" in output
    assert "Official facility page: https://example.nyc/restroom {cite:S1}" in output
    assert "Map: https://www.google.com/maps/search/" in output
    assert "query=40.00100,-73.00000 {cite:S1}" in output
    assert "Current entry and restroom fixture condition were not verified" in output
    assert "report the problem to 311" not in output
    assert "records are not real-time" not in output
    for label in (
        "Seasonal availability",
        "NYC listing accessibility",
        "Restroom type",
        "Changing station",
        "Access note",
    ):
        line = next(line for line in output.splitlines() if label in line)
        assert line.endswith("{cite:S1}")


@pytest.mark.asyncio
async def test_lookup_honors_requested_access_and_changing_station_filters(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "near",
                "facility_name": "Nearest Restroom",
                "latitude": "40.0001",
                "longitude": "-73.0",
                "status": "Operational",
                "accessibility": "Not Accessible",
                "changing_stations": "No",
            },
            {
                ":id": "usable",
                "facility_name": "Usable Restroom",
                "latitude": "40.001",
                "longitude": "-73.0",
                "status": "Operational",
                "accessibility": "Fully Accessible",
                "changing_stations": "Yes, in single-stall all gender restroom only",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    filtered = await restrooms.get_tools()[0].handler(
        {"near": "here", "fully_accessible": True, "changing_station": True}, ctx
    )
    unfiltered = await restrooms.get_tools()[0].handler({"near": "here", "max_results": 1}, ctx)

    assert "Usable Restroom" in filtered
    assert "Nearest Restroom" not in filtered
    assert "does not prove restroom fixture accessibility" in filtered
    assert "City-dataset matches for site accessibility and a changing station" in filtered
    assert "matches for every requested feature" not in filtered
    assert "311" not in filtered
    assert "1. Nearest Restroom" in unfiltered


@pytest.mark.asyncio
async def test_lookup_uses_named_city_record_and_keeps_closer_partial_match(monkeypatch):
    fetched = []

    async def missing_geocode(text, **kwargs):
        return None

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "origin",
                "facility_name": "Neighborhood Plaza APT",
                "latitude": "40.0000",
                "longitude": "-73.0000",
                "status": "Operational",
                "accessibility": "Fully Accessible",
                "changing_stations": "No",
            },
            {
                ":id": "partial",
                "facility_name": "Neighborhood Library",
                "latitude": "40.0010",
                "longitude": "-73.0000",
                "status": "Operational",
                "accessibility": "",
                "changing_stations": "Yes",
                "website": {"url": "https://library.example/branch"},
            },
            {
                ":id": "complete",
                "facility_name": "Far Library",
                "latitude": "40.1000",
                "longitude": "-73.0000",
                "status": "Operational",
                "accessibility": "Fully Accessible",
                "changing_stations": "Yes",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        return []

    async def fake_web_fetch(args, _ctx):
        fetched.append(args)
        return "SOURCE S99: Official page\nWheelchair accessible\n{cite:S99}"

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", missing_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry.discover(config.MODULES_DIR),
        toolbox={"web_fetch": SimpleNamespace(handler=fake_web_fetch)},
    )

    output = await restrooms.get_tools()[0].handler(
        {
            "near": "Neighborhood Plaza, Queens, NY",
            "fully_accessible": True,
            "changing_station": True,
        },
        ctx,
    )

    assert "Neighborhood Plaza APT" in output
    assert "Far Library" in output
    assert "Closest supported match" in output
    assert "Neighborhood Library" in output
    assert "Missing constraint: site accessibility is not listed" not in output
    assert "https://library.example/branch" in output
    assert "Official page resolves the missing site-accessibility field" in output
    assert fetched == [{
        "url": "https://library.example/branch",
        "find": "site wheelchair accessibility",
    }]
    assert output.index("- Neighborhood Library") < output.index("- Neighborhood Plaza APT")
    assert output.index("Neighborhood Library") < output.index("Far Library")


@pytest.mark.asyncio
async def test_lookup_uses_the_residents_requested_date(monkeypatch):
    seen_where = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text, match_type="coordinates")

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "friday",
                "facility_name": "Friday Plaza Restroom",
                "latitude": "40.0001",
                "longitude": "-73.0",
                "status": "Operational",
            },
            {
                ":id": "saturday",
                "facility_name": "Saturday Plaza Restroom",
                "latitude": "40.001",
                "longitude": "-73.0",
                "status": "Operational",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        seen_where.append(kwargs["where"])
        return [
            {
                "OBJECTID": 1,
                "Facility_name": "Friday Plaza Restroom",
                "lat": 40.0001,
                "lon": -73.0,
                "Finder_status": "OPEN",
                "Friday": "9a-5p",
                "cc_fri_open1": "09:00 AM",
                "cc_fri_close1": "05:00 PM",
            },
            {
                "OBJECTID": 2,
                "Facility_name": "Saturday Plaza Restroom",
                "lat": 40.001,
                "lon": -73.0,
                "Finder_status": "OPEN",
                "Saturday": "10a-2p",
                "cc_sat_open1": "10:00 AM",
                "cc_sat_close1": "02:00 PM",
            },
        ]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 7, 17, 12, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler(
        {"near": "here", "max_results": 1, "visit_date": "2026-07-18"},
        ctx,
    )

    assert "1. Saturday Plaza Restroom" in output
    assert "1. Saturday Plaza Restroom, 0.07 miles {cite:" in output
    assert "straight-line" not in output
    assert "Friday Plaza Restroom" not in output
    assert "site building is scheduled on Saturday, 2026-07-18" in output
    assert "scheduled open now" not in output
    assert (
        "Ranked by requested-day schedule evidence, then distance, not by longest hours "
        "or restroom quality."
    ) in output
    limitation = next(
        line
        for line in output.splitlines()
        if line.startswith("Current entry and restroom fixture condition")
    )
    assert "{cite:" not in limitation
    assert seen_where == ["1=1"]


@pytest.mark.asyncio
async def test_lookup_applies_an_explicit_visit_time_before_distance(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [
            {
                ":id": "near",
                "facility_name": "Morning Plaza Restroom",
                "latitude": "40.0001",
                "longitude": "-73.0",
                "status": "Operational",
            },
            {
                ":id": "far",
                "facility_name": "Afternoon Plaza Restroom",
                "latitude": "40.001",
                "longitude": "-73.0",
                "status": "Operational",
            },
        ]

    async def fake_cool_options(*args, **kwargs):
        return [
            {
                "Facility_name": "Morning Plaza Restroom",
                "lat": 40.0001,
                "lon": -73.0,
                "cc_fri_open1": "08:00 AM",
                "cc_fri_close1": "10:00 AM",
            },
            {
                "Facility_name": "Afternoon Plaza Restroom",
                "lat": 40.001,
                "lon": -73.0,
                "cc_fri_open1": "11:00 AM",
                "cc_fri_close1": "04:00 PM",
            },
        ]

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 8, 21, 9, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler(
        {
            "near": "here",
            "max_results": 1,
            "visit_date": "2026-08-21",
            "visit_time": "12:00",
        },
        ctx,
    )

    assert "1. Afternoon Plaza Restroom" in output
    assert "Morning Plaza Restroom" not in output
    assert "scheduled open at 12:00 PM" in output


@pytest.mark.asyncio
async def test_future_lookup_does_not_describe_uncorroborated_result_as_open_now(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [{
            ":id": "1",
            "facility_name": "Uncorroborated Restroom",
            "latitude": "40.001",
            "longitude": "-73.0",
            "status": "Operational",
            "hours_of_operation": "Saturday 10am-2pm",
        }]

    async def fake_cool_options(*args, **kwargs):
        return []

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 7, 17, 12, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler(
        {"near": "here", "visit_date": "2026-07-18"},
        ctx,
    )

    assert "future building schedule is not independently corroborated" in output
    assert "open now" not in output


@pytest.mark.asyncio
async def test_lookup_rejects_past_date_before_external_calls(monkeypatch):
    monkeypatch.setattr(
        restrooms,
        "_nyc_now",
        lambda: datetime(2026, 7, 17, 12, 0, tzinfo=ZoneInfo("America/New_York")),
    )

    async def should_not_geocode(*args, **kwargs):
        raise AssertionError("past date reached geocoder")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", should_not_geocode)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler(
        {"near": "here", "visit_date": "2026-07-16"},
        ctx,
    )

    assert "cannot verify a past service date" in output


@pytest.mark.asyncio
async def test_lookup_asks_for_a_better_location_when_geocoding_fails(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return None

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    registry = Registry.discover(config.MODULES_DIR)
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    output = await restrooms.get_tools()[0].handler({"near": "somewhere"}, ctx)

    assert "specific NYC address or landmark" in output


@pytest.mark.asyncio
async def test_lookup_exposes_geocoder_provider_outage(monkeypatch) -> None:
    async def unavailable(*args, **kwargs):
        raise GeocoderUnavailable

    async def city_rows(*args, **kwargs):
        return SimpleNamespace(
            records=[{
                ":id": "1",
                "facility_name": "City Restroom",
                "latitude": "40.001",
                "longitude": "-73.0",
                "status": "Operational",
            }],
            complete=True,
        )

    async def cool_rows(*args, **kwargs):
        return SimpleNamespace(records=[], complete=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", unavailable)
    monkeypatch.setattr(restrooms, "query_dataset_pages", city_rows)
    monkeypatch.setattr(restrooms, "query_feature_service", cool_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].invoke({"near": "somewhere"}, ctx)

    assert isinstance(output, ToolFailure)
    assert output.status == "unavailable"


@pytest.mark.asyncio
async def test_lookup_exposes_geocoder_outage_when_city_feed_is_empty(monkeypatch) -> None:
    async def unavailable(*args, **kwargs):
        raise GeocoderUnavailable

    async def city_rows(*args, **kwargs):
        return SimpleNamespace(records=[], complete=True)

    async def cool_rows(*args, **kwargs):
        return SimpleNamespace(records=[], complete=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", unavailable)
    monkeypatch.setattr(restrooms, "query_dataset_pages", city_rows)
    monkeypatch.setattr(restrooms, "query_feature_service", cool_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].invoke({"near": "somewhere"}, ctx)

    assert isinstance(output, ToolFailure)
    assert output.status == "unavailable"


@pytest.mark.asyncio
async def test_lookup_propagates_unexpected_geocoder_error_with_exact_dataset_match(
    monkeypatch,
) -> None:
    async def broken(*args, **kwargs):
        raise RuntimeError("programming error")

    async def city_rows(*args, **kwargs):
        return SimpleNamespace(
            records=[{
                ":id": "1",
                "facility_name": "City Restroom",
                "latitude": "40.001",
                "longitude": "-73.0",
                "status": "Operational",
            }],
            complete=True,
        )

    async def cool_rows(*args, **kwargs):
        return SimpleNamespace(records=[], complete=True)

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", broken)
    monkeypatch.setattr(restrooms, "query_dataset_pages", city_rows)
    monkeypatch.setattr(restrooms, "query_feature_service", cool_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    with pytest.raises(RuntimeError, match="programming error"):
        await restrooms.get_tools()[0].invoke({"near": "City Restroom"}, ctx)


@pytest.mark.asyncio
async def test_lookup_keeps_city_results_when_cool_options_is_unavailable(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.0, -73.0, text)

    async def fake_city(*args, **kwargs):
        return [{
            ":id": "1", "facility_name": "City Restroom", "latitude": "40.001",
            "longitude": "-73.0", "status": "Operational",
        }]

    async def failed_cool_options(*args, **kwargs):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr("heynyc.core.tools.geo.geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", failed_cool_options)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await restrooms.get_tools()[0].handler({"near": "here"}, ctx)

    assert "City Restroom" in output
    assert "Cool Options cross-check was unavailable" in output


def test_nearby_unrelated_cool_option_is_not_same_site_corroboration():
    place = type("Place", (), {"name": "53rd Street Library", "lat": 40.0, "lon": -73.0})()
    records = [{
        "Facility_name": "Unrelated Coffee Shop", "lat": 40.0001, "lon": -73.0,
    }]

    assert restrooms._matching_cool_option(place, records) is None


def test_nearby_sub_business_is_not_same_site_corroboration():
    place = type("Place", (), {"name": "Central Library", "lat": 40.0, "lon": -73.0})()
    records = [{
        "Facility_name": "Central Library Cafe", "lat": 40.0001, "lon": -73.0,
    }]

    assert restrooms._matching_cool_option(place, records) is None


def test_restroom_schedule_handles_overnight_hours():
    record = {"cc_wed_open1": "08:00 PM", "cc_wed_close1": "02:00 AM"}
    now = datetime(2026, 7, 16, 1, 0, tzinfo=ZoneInfo("America/New_York"))

    assert restrooms._open_now(record, now) is True
