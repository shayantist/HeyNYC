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


@pytest.mark.asyncio
async def test_lookup_flags_closer_centers_closed_now_with_reopening(monkeypatch):
    # F068: 8:30 PM Saturday, the only open center is a Petco 2 miles away while
    # closer library/senior centers are closed. The output must state, in data
    # terms, how many closer centers are closed now and the soonest reopening.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            return []
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC_PETCO",
                "FACILITY_NAME": "Petco 86th Lexington",
                "ADDRESS": "147 E 86TH ST",
                "lat": 40.7870,  # ~2 miles north of origin
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "cc_sat_open1": "09:00 AM",
                "cc_sat_close1": "09:00 PM",
            },
            {
                "OBJECTID": 2,
                "NYCEM_ID": "CC_LIB",
                "FACILITY_NAME": "Morningside Library",
                "ADDRESS": "2900 BROADWAY",
                "lat": 40.7609,  # ~0.2 miles away
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "cc_sat_open1": "10:00 AM",
                "cc_sat_close1": "05:00 PM",
                "cc_mon_open1": "09:00 AM",  # reopens Monday
                "cc_mon_close1": "05:00 PM",
            },
            {
                "OBJECTID": 3,
                "NYCEM_ID": "CC_SR",
                "FACILITY_NAME": "Hamilton Senior Center",
                "ADDRESS": "141 W 140TH ST",
                "lat": 40.7623,  # ~0.3 miles away
                "lon": -73.9780,
                "Finder_status": "OPEN",
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
    # senior centers with no all-ages option. Rows the dataset itself types "Older Adult
    # Center" carry their restriction as language-independent data the model can translate,
    # and when such rows dominate the shown results the tool surfaces an all-ages option it
    # can cite (here the library), plus the pools/spray-showers/libraries category note.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            return []
        return [
            {"OBJECTID": 1, "NYCEM_ID": "CC_OA1", "FACILITY_NAME": "Carter Older Adult Center",
             "ADDRESS": "1 E 100 ST", "lat": 40.7595, "lon": -73.9780, "Finder_status": "OPEN",
             "FACILITY_TYPE": "Older Adult Center",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "05:00 PM"},
            {"OBJECTID": 2, "NYCEM_ID": "CC_OA2", "FACILITY_NAME": "Dyckman Older Adult Center",
             "ADDRESS": "2 E 100 ST", "lat": 40.7600, "lon": -73.9780, "Finder_status": "OPEN",
             "FACILITY_TYPE": "Older Adult Center",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "05:00 PM"},
            {"OBJECTID": 3, "NYCEM_ID": "CC_LIB", "FACILITY_NAME": "Morningside Library",
             "ADDRESS": "2900 BROADWAY", "lat": 40.7640, "lon": -73.9780, "Finder_status": "OPEN",
             "FACILITY_TYPE": "Library",
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

    # The two nearest are older-adult centers; each carries its declared restriction as data.
    assert "Older Adult Center" in output
    assert "age-restricted" in output.lower()
    # They dominate the shown results, so an all-ages lane is surfaced and cited (the library),
    # with the general all-ages category note.
    assert "all ages" in output.lower()
    assert "nearest all-ages option" in output.lower()
    assert "Morningside Library" in output


@pytest.mark.asyncio
async def test_all_ages_results_get_no_restriction_note(monkeypatch):
    # F072 inverse (the fence on the other side): all-ages results (libraries) get no
    # age-restriction annotation and no older-adult steering note.
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        if url == cooling.COOL_OPTIONS_URL:
            return []
        return [
            {"OBJECTID": 1, "NYCEM_ID": "L1", "FACILITY_NAME": "Aguilar Library",
             "ADDRESS": "1 E 110 ST", "lat": 40.7595, "lon": -73.9780, "Finder_status": "OPEN",
             "FACILITY_TYPE": "Library",
             "cc_wed_open1": "09:00 AM", "cc_wed_close1": "08:00 PM"},
            {"OBJECTID": 2, "NYCEM_ID": "L2", "FACILITY_NAME": "Harlem Library",
             "ADDRESS": "9 W 124 ST", "lat": 40.7600, "lon": -73.9780, "Finder_status": "OPEN",
             "FACILITY_TYPE": "Library",
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
