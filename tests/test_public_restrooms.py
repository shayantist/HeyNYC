"""Public-restroom lookup chooses useful, corroborated options."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.public_restrooms import tools as restrooms


def test_public_restrooms_module_loads_custom_lookup():
    registry = Registry.discover(config.MODULES_DIR)

    tool_names = {tool.name for tool in registry.load_module_tools()}

    assert "public_restroom_lookup" in tool_names


@pytest.mark.asyncio
async def test_lookup_prefers_open_official_corroboration_and_respects_limit(monkeypatch):
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

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
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
        {"near": "Rockefeller Center", "limit": 1}, ctx
    )

    assert "1. 645 Fifth Avenue POPS" in output
    assert "Unverified Lobby" not in output
    assert "scheduled open now" in output
    assert "does not confirm the restroom's condition" in output
    assert "Wednesday: 8a-10p" in output
    assert "Wheelchair accessible: Yes" in output
    assert "NYC restroom record date: 2025-11-05" in output
    assert "{cite:S1}" in output
    assert "{cite:S2}" in output
    assert len(ctx.citations.mapping()) == 2


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

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    registry = Registry.discover(config.MODULES_DIR)
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    output = await restrooms.get_tools()[0].handler({"near": "here", "limit": 99}, ctx)

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

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
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
    assert "Official facility page: https://example.nyc/restroom" in output


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

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
    monkeypatch.setattr(restrooms, "query_dataset", fake_city)
    monkeypatch.setattr(restrooms, "query_feature_service", fake_cool_options)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    filtered = await restrooms.get_tools()[0].handler(
        {"near": "here", "fully_accessible": True, "changing_station": True}, ctx
    )
    unfiltered = await restrooms.get_tools()[0].handler({"near": "here", "limit": 1}, ctx)

    assert "Usable Restroom" in filtered
    assert "Nearest Restroom" not in filtered
    assert "1. Nearest Restroom" in unfiltered


@pytest.mark.asyncio
async def test_lookup_asks_for_a_better_location_when_geocoding_fails(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return None

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
    registry = Registry.discover(config.MODULES_DIR)
    ctx = ToolContext(citations=CitationRegistry(), registry=registry)

    output = await restrooms.get_tools()[0].handler({"near": "somewhere"}, ctx)

    assert "specific NYC address or landmark" in output


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

    monkeypatch.setattr(restrooms, "geocode", fake_geocode)
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
