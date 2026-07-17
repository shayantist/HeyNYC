"""Cool Options lookup uses the current City finder feeds."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.cooling_centers import tools as cooling


def test_cooling_module_loads_current_cool_options_lookup():
    registry = Registry.discover(config.MODULES_DIR)

    tool_names = {tool.name for tool in registry.load_module_tools()}

    assert "cool_options_lookup" in tool_names


@pytest.mark.asyncio
async def test_lookup_combines_active_centers_and_indoor_cool_options(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            return [
                {
                    "OBJECTID": 17439,
                    "NYCEM_ID": "CO016",
                    "Facility_name": "POPS - 645 Fifth Avenue",
                    "Address": "645 5TH AVENUE",
                    "lat": 40.7592,
                    "lon": -73.9761,
                    "Finder_status": "OPEN",
                    "Location_type": "Indoor",
                    "Space_type": "Other Indoor Cool Option",
                    "Accessible": "Yes",
                    "Pet_friendly": "No",
                    "Wednesday": "8a-10p",
                    "cc_wed_open1": "08:00 AM",
                    "cc_wed_close1": "10:00 PM",
                }
            ]
        return [
            {
                "OBJECTID": 2880,
                "NYCEM_ID": "CC1043",
                "FACILITY_NAME": "Petco Turtle Bay",
                "ADDRESS": "991 2 Ave",
                "lat": 40.7569,
                "lon": -73.9677,
                "Finder_status": "OPEN",
                "FACILITY_TYPE": "Other",
                "Accessible": "Yes",
                "PET_FRIENDLY": "Yes",
                "Wednesday": "9a-9p",
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "09:00 PM",
            }
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 13, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Rockefeller Center", "kind": "all", "limit": 2}, ctx
    )

    assert "POPS - 645 Fifth Avenue" in output
    assert "other indoor cool option" in output
    assert "Petco Turtle Bay" in output
    assert "activated cooling center" in output
    assert "scheduled open now" in output
    assert "Resolved 'Rockefeller Center'" in output
    assert "Wednesday: 8a-10p" in output
    assert "Accessible: Yes" in output
    assert output.count("https://www.google.com/maps/search/?api=1&query=") == 2
    assert len(ctx.citations.mapping()) == 2


@pytest.mark.asyncio
async def test_lookup_can_return_only_activated_cooling_centers(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            return [{"Facility_name": "Indoor Atrium", "lat": 40.7581, "lon": -73.9780}]
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC1",
                "FACILITY_NAME": "Active Center",
                "lat": 40.7600,
                "lon": -73.9780,
                "Finder_status": "OPEN",
            }
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Rockefeller Center", "kind": "cooling_center"}, ctx
    )

    assert "Active Center" in output
    assert "Indoor Atrium" not in output


@pytest.mark.asyncio
async def test_lookup_reports_when_no_cooling_centers_are_activated(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return []

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Rockefeller Center", "kind": "cooling_center"}, ctx
    )

    assert "No activated cooling centers" in output


@pytest.mark.asyncio
async def test_lookup_keeps_indoor_options_when_active_feed_is_unavailable(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.ACTIVE_CENTERS_URL:
            raise RuntimeError("source unavailable")
        return [{
            "OBJECTID": 2,
            "NYCEM_ID": "CO2",
            "Facility_name": "Indoor Atrium",
            "lat": 40.7581,
            "lon": -73.9780,
            "Location_type": "Indoor",
            "Space_type": "Other Indoor Cool Option",
        }]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Rockefeller Center", "kind": "all"}, ctx
    )

    assert "Indoor Atrium" in output
    assert "activated-center feed was unavailable" in output


def test_cooling_schedule_handles_overnight_hours():
    record = {"cc_wed_open1": "08:00 PM", "cc_wed_close1": "02:00 AM"}
    now = datetime(2026, 7, 16, 1, 0, tzinfo=ZoneInfo("America/New_York"))

    assert cooling._open_now(record, now) is True


@pytest.mark.asyncio
async def test_lookup_labels_general_feed_failure_when_active_feed_is_empty(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            raise RuntimeError("source unavailable")
        return []

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler({"near": "Rockefeller Center"}, ctx)

    assert "general Cool Options feed was unavailable" in output
    assert "No matching NYC Cool Options were found" not in output


def test_cooling_schedule_checks_second_interval():
    record = {
        "cc_wed_open1": "08:00 AM",
        "cc_wed_close1": "10:00 AM",
        "cc_wed_open2": "05:00 PM",
        "cc_wed_close2": "09:00 PM",
    }
    now = datetime(2026, 7, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))

    assert cooling._open_now(record, now) is True
