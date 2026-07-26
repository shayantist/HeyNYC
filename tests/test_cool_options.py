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
    calls = []

    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        calls.append((url, kwargs["where"]))
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
            },
            {
                "OBJECTID": 2880,
                "NYCEM_ID": "CC1043",
                "Facility_name": "Petco Turtle Bay",
                "Address": "991 2 Ave",
                "lat": 40.7569,
                "lon": -73.9677,
                "Finder_status": "OPEN",
                "Location_type": "Indoor",
                "Space_type": "Cooling Center",
                "Accessible": "Yes",
                "Pet_friendly": "Yes",
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
    assert calls == [(cooling.COOL_OPTIONS_URL, "Finder_status='OPEN'")]


@pytest.mark.asyncio
async def test_lookup_can_return_only_activated_cooling_centers(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {
                "Facility_name": "Indoor Atrium",
                "lat": 40.7581,
                "lon": -73.9780,
                "Space_type": "Other Indoor Cool Option",
            },
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC1",
                "Facility_name": "Active Center",
                "lat": 40.7600,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
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
async def test_lookup_reports_when_finder_is_unavailable(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Rockefeller Center", "kind": "all"}, ctx
    )

    assert "NYC Cool Options finder was unavailable" in output


def test_cooling_schedule_handles_overnight_hours():
    record = {"cc_wed_open1": "08:00 PM", "cc_wed_close1": "02:00 AM"}
    now = datetime(2026, 7, 16, 1, 0, tzinfo=ZoneInfo("America/New_York"))

    assert cooling._open_now(record, now) is True


def test_cooling_schedule_checks_second_interval():
    record = {
        "cc_wed_open1": "08:00 AM",
        "cc_wed_close1": "10:00 AM",
        "cc_wed_open2": "05:00 PM",
        "cc_wed_close2": "09:00 PM",
    }
    now = datetime(2026, 7, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))

    assert cooling._open_now(record, now) is True


@pytest.mark.asyncio
async def test_lookup_flags_closer_centers_closed_now_with_reopening(monkeypatch):
    # F068: 8:30 PM Saturday, the only open center is a Petco 2 miles away while
    # closer library/senior centers are closed. The output must state, in data
    # terms, how many closer centers are closed now and the soonest reopening.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC_PETCO",
                "Facility_name": "Petco 86th Lexington",
                "Address": "147 E 86TH ST",
                "lat": 40.7870,  # ~2 miles north of origin
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_sat_open1": "09:00 AM",
                "cc_sat_close1": "09:00 PM",
            },
            {
                "OBJECTID": 2,
                "NYCEM_ID": "CC_LIB",
                "Facility_name": "Morningside Library",
                "Address": "2900 BROADWAY",
                "lat": 40.7609,  # ~0.2 miles away
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_sat_open1": "10:00 AM",
                "cc_sat_close1": "05:00 PM",
                "cc_mon_open1": "09:00 AM",  # reopens Monday
                "cc_mon_close1": "05:00 PM",
            },
            {
                "OBJECTID": 3,
                "NYCEM_ID": "CC_SR",
                "Facility_name": "Hamilton Senior Center",
                "Address": "141 W 140TH ST",
                "lat": 40.7623,  # ~0.3 miles away
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_sat_open1": "09:00 AM",
                "cc_sat_close1": "04:00 PM",
                "cc_sun_open1": "09:00 AM",  # reopens Sunday, the soonest
                "cc_sun_close1": "05:00 PM",
            },
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 18, 20, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Columbia University", "kind": "cooling_center", "limit": 1}, ctx
    )

    assert "Petco 86th Lexington" in output
    # Only Petco is listed (limit 1), so the summary line must carry the rest.
    assert "Morningside Library" not in output
    assert "2 closer" in output
    assert "closed right now" in output
    assert "Sunday 09:00 AM" in output


@pytest.mark.asyncio
async def test_older_adult_centers_annotated_and_all_ages_note(monkeypatch):
    # F072: a parent asking where to take kids must not be handed only "older adults only"
    # senior centers. Rows the dataset itself marks age-restricted carry their restriction
    # as language-independent data the model can translate, and when such rows dominate the
    # shown results the tool surfaces a cited option not marked age-restricted.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {"OBJECTID": 1, "NYCEM_ID": "CC_OA1", "Facility_name": "Carter Older Adult Center",
             "Address": "1 E 100 ST", "lat": 40.7595, "lon": -73.9780, "Finder_status": "OPEN",
             "Space_type": "Cooling Center", "Age_restriction": "Yes",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "05:00 PM"},
            {"OBJECTID": 2, "NYCEM_ID": "CC_OA2", "Facility_name": "Dyckman Older Adult Center",
             "Address": "2 E 100 ST", "lat": 40.7600, "lon": -73.9780, "Finder_status": "OPEN",
             "Space_type": "Cooling Center", "Age_restriction": "Yes",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "05:00 PM"},
            {"OBJECTID": 3, "NYCEM_ID": "CC_LIB", "Facility_name": "Morningside Library",
             "Address": "2900 BROADWAY", "lat": 40.7640, "lon": -73.9780, "Finder_status": "OPEN",
             "Space_type": "Cooling Center", "Age_restriction": "No",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "08:00 PM"},
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
        {
            "near": "Central Park",
            "kind": "cooling_center",
            "audience": "not_age_restricted",
            "limit": 2,
        },
        ctx,
    )

    assert "Carter Older Adult Center" not in output
    assert "Dyckman Older Adult Center" not in output
    assert "City row is not marked age-restricted" in output
    assert "pools" not in output.lower()
    assert "spray showers" not in output.lower()
    assert "Morningside Library" in output
    assert (
        cooling.get_tools()[0].parameters["properties"]["audience"]["enum"]
        == ["any", "not_age_restricted"]
    )


@pytest.mark.asyncio
async def test_all_ages_results_get_no_restriction_note(monkeypatch):
    # F072 inverse (the fence on the other side): all-ages results (libraries) get no
    # age-restriction annotation and no older-adult steering note.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {"OBJECTID": 1, "NYCEM_ID": "L1", "Facility_name": "Aguilar Library",
             "Address": "1 E 110 ST", "lat": 40.7595, "lon": -73.9780, "Finder_status": "OPEN",
             "Space_type": "Cooling Center", "Age_restriction": "No",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "08:00 PM"},
            {"OBJECTID": 2, "NYCEM_ID": "L2", "Facility_name": "Harlem Library",
             "Address": "9 W 124 ST", "lat": 40.7600, "lon": -73.9780, "Finder_status": "OPEN",
             "Space_type": "Cooling Center", "Age_restriction": "No",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "08:00 PM"},
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
        {"near": "Central Park", "kind": "cooling_center", "limit": 2}, ctx
    )

    assert "age-restricted" not in output.lower()
    assert "all-ages option" not in output.lower()


def test_cooling_next_open_skips_todays_passed_intervals():
    record = {
        "cc_sat_open1": "09:00 AM",
        "cc_sat_close1": "04:00 PM",
        "cc_sun_open1": "09:00 AM",
    }
    now = datetime(2026, 7, 18, 20, 30, tzinfo=ZoneInfo("America/New_York"))  # Saturday

    assert cooling._next_open(record, now) == (1, 540, "Sunday 09:00 AM")


@pytest.mark.asyncio
async def test_lookup_uses_requested_date_instead_of_current_day(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CLOSER",
                "Facility_name": "Closer Friday Library",
                "Address": "1 Main St",
                "lat": 40.7581,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "Friday": "10a-6p",
                "Saturday": "CLOSED",
                "cc_fri_open1": "10:00 AM",
                "cc_fri_close1": "06:00 PM",
            },
            {
                "OBJECTID": 2,
                "NYCEM_ID": "SATURDAY",
                "Facility_name": "Saturday Library",
                "Address": "2 Main St",
                "lat": 40.7590,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_sat_open1": "10:00 AM",
                "cc_sat_close1": "05:00 PM",
            },
            {
                "OBJECTID": 3,
                "NYCEM_ID": "UNKNOWN",
                "Facility_name": "Unknown Saturday Hours",
                "Address": "3 Main St",
                "lat": 40.7585,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
            },
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 24, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry.discover(config.MODULES_DIR)
    )

    output = await cooling.get_tools()[0].handler(
        {
            "near": "Flushing, Queens",
            "kind": "cooling_center",
            "limit": 2,
            "on": "2026-07-25",
        },
        ctx,
    )

    assert "Saturday Library" in output
    assert "Unknown Saturday Hours" in output
    assert "Closer Friday Library" not in output
    assert "Saturday, July 25, 2026: 10:00 AM-05:00 PM" in output
    assert "Activation status is current at lookup time" in output
    assert (
        "Activation: current at lookup only; not verified for Saturday, July 25, 2026"
        in output
    )
    assert "Saturday Library, activated cooling center" not in output
    assert "one-off closures" in output
    assert "scheduled open now" not in output

    schema = cooling.get_tools()[0].parameters
    assert schema["properties"]["on"]["format"] == "date"
    assert "on" not in schema["required"]
