"""Cool Options lookup uses the current City finder feeds."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ResidentFact, ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.cooling_centers import tools as cooling


def test_cooling_module_loads_current_find_cool_options():
    registry = Registry.discover(config.MODULES_DIR)

    tool_names = {tool.name for tool in registry.load_module_tools()}

    assert "find_cool_options" in tool_names


def _site_row(key: str, name: str, object_id: int = 1, lat: float = 40.7600) -> dict:
    return {
        "OBJECTID": object_id,
        "NYCEM_ID": key.upper(),
        "Facility_name": name,
        "Address": "123 W 42 ST",
        "lat": lat,
        "lon": -73.9780,
        "Finder_status": "OPEN",
        "Space_type": "Cooling Center",
    }


def _site_rows() -> list[dict]:
    return [
        _site_row("raices", "Raices Times Square"),
        _site_row("other", "Closer Cooling Site", 2, 40.7581),
    ]


_KNOWN_WEDNESDAY_HOURS = {
    "cc_wed_open1": "09:00 AM",
    "cc_wed_close1": "05:00 PM",
}


def _context(turn: str, facts: dict | None = None) -> ToolContext:
    return ToolContext(
        citations=CitationRegistry(),
        registry=Registry.discover(config.MODULES_DIR),
        user_turns=(turn,),
        resident_facts=facts or {},
    )


def _patch_lookup(monkeypatch, rows: list[dict]):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return rows

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    return cooling.get_tools()[0].handler


@pytest.fixture
def lookup(monkeypatch):
    rows = [{**row, **_KNOWN_WEDNESDAY_HOURS} for row in _site_rows()]
    return _patch_lookup(monkeypatch, rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("near", ["Flushing", "Queens", "Brooklyn", "New York"])
async def test_origin_only_tokens_do_not_select_a_facility(monkeypatch, near):
    lookup = _patch_lookup(
        monkeypatch,
        [_site_row("origin", f"{near} Library"), _site_row("other", "Other Center", 2)],
    )
    ctx = _context(f"options near {near}")

    await lookup({"near": near, "kind": "cooling_center"}, ctx)

    assert "/cooling/site" not in ctx.resident_facts


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Site Times Square", "Cooling Center 7"])
async def test_exact_full_generic_leading_facility_name_is_selectable(monkeypatch, name):
    lookup = _patch_lookup(
        monkeypatch,
        [
            {
                **_site_row("chosen", name),
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "05:00 PM",
            },
            {
                **_site_row("other", "Other Center", 2, 40.7581),
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "05:00 PM",
            },
        ],
    )
    ctx = _context(f"Please take me to {name}")

    output = await lookup(
        {"near": "Flushing, Queens", "kind": "cooling_center", "site": name}, ctx
    )

    assert output.splitlines()[2].startswith(f"1. {name}")
    assert ctx.resident_facts["/cooling/site"].value["key"] == "chosen"


@pytest.mark.asyncio
async def test_resident_description_does_not_select_a_facility(monkeypatch):
    lookup = _patch_lookup(
        monkeypatch,
        [
            {
                **_site_row("senior", "Senior Center at St Peter's", lat=40.8500),
                **_KNOWN_WEDNESDAY_HOURS,
            },
            {
                **_site_row("library", "Brooklyn Central Library", 2, 40.7581),
                **_KNOWN_WEDNESDAY_HOURS,
            },
        ],
    )
    ctx = _context("I'm a senior using a walker. Where can I cool down?")

    output = await lookup({"near": "Crown Heights", "kind": "all"}, ctx)

    assert output.splitlines()[2].startswith("1. Brooklyn Central Library")
    assert "/cooling/site" not in ctx.resident_facts


def test_site_selection_requires_the_exact_tool_argument():
    items = [
        {"name": "Raices Times Square"},
        {"name": "Closer Cooling Site"},
    ]

    assert cooling._site_from_turn(
        items, "What are its hours?", requested="Raices Times Square"
    ) == items[0]
    assert cooling._site_from_turn(items, "Raices", requested="Raices") is None
    assert cooling._site_from_turn(items, "I'm a senior") is None


@pytest.mark.asyncio
async def test_later_selection_is_limited_to_the_stored_offered_set(lookup):
    ctx = _context(
        "Closer Cooling Site",
        {
            "/cooling/offered": ResidentFact(
                value={
                    "keys": ["raices"],
                    "origin": [40.7580, -73.9780],
                    "scope": {"kind": "cooling_center", "audience": "any"},
                },
                source_turn_id="1",
                status="captured",
            ),
            "/cooling/site": ResidentFact(
                value={"key": "raices", "origin": [40.7580, -73.9780]},
                source_turn_id="1",
                status="captured",
            ),
        },
    )

    output = await lookup(
        {"near": "Flushing, Queens", "kind": "cooling_center", "site": "Closer Cooling Site"},
        ctx,
    )

    assert "1. Raices Times Square" in output
    assert "Closer Cooling Site" not in output
    assert ctx.resident_facts["/cooling/site"].value["key"] == "raices"


@pytest.mark.asyncio
async def test_origin_change_replaces_offered_and_selected_state(monkeypatch):
    async def fake_geocode(text, **kwargs):
        if text == "Flushing, Queens":
            return GeoPoint(40.7580, -73.9780, text)
        return GeoPoint(40.7000, -73.9000, text)

    calls = 0

    async def fake_query(url, **kwargs):
        nonlocal calls
        calls += 1
        return _site_rows() if calls == 1 else [_site_row("brooklyn", "Brooklyn Center")]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    ctx = _context("Raices")
    await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"}, ctx
    )
    ctx.user_turns = ("Show me options near Brooklyn",)

    await cooling.get_tools()[0].handler(
        {"near": "Brooklyn", "kind": "cooling_center"}, ctx
    )

    assert ctx.resident_facts["/cooling/offered"].value == {
        "keys": ["brooklyn"],
        "origin": [40.7, -73.9],
        "scope": {"kind": "cooling_center", "audience": "any"},
    }
    assert "/cooling/site" not in ctx.resident_facts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_args", "expected_scope"),
    [
        ({"kind": "indoor", "audience": "any"}, {"kind": "indoor", "audience": "any"}),
        (
            {"kind": "cooling_center", "audience": "not_age_restricted"},
            {"kind": "cooling_center", "audience": "not_age_restricted"},
        ),
    ],
)
async def test_same_origin_scope_change_replaces_offered_and_selected_state(
    monkeypatch, second_args, expected_scope
):
    rows = [
        {
            **_site_row("old", "Old Cooling Center"),
            "Location_type": "Outdoor",
            "Age_restriction": "Yes",
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "05:00 PM",
        },
        {
            **_site_row("new", "New Filtered Option", 2, 40.7581),
            "Location_type": "Indoor",
            "Age_restriction": "No",
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "05:00 PM",
        },
    ]
    lookup = _patch_lookup(monkeypatch, rows)
    ctx = _context("Old Cooling Center")

    await lookup(
        {
            "near": "Flushing, Queens",
            "kind": "cooling_center",
            "audience": "any",
            "site": "Old Cooling Center",
        },
        ctx,
    )
    assert ctx.resident_facts["/cooling/site"].value["key"] == "old"
    ctx.user_turns = ("Show me options",)

    output = await lookup(
        {"near": "Flushing, Queens", **second_args},
        ctx,
    )

    assert "New Filtered Option" in output
    assert "I couldn't re-confirm" not in output
    assert ctx.resident_facts["/cooling/offered"].value == {
        "keys": ["new"],
        "origin": [40.758, -73.978],
        "scope": expected_scope,
    }
    assert "/cooling/site" not in ctx.resident_facts


@pytest.mark.asyncio
@pytest.mark.parametrize("path,value", [
    ("/cooling/site", None),
    ("/cooling/offered", {"keys": "raices", "origin": [40.7580, -73.9780]}),
])
async def test_malformed_cooling_state_is_cleared_without_raising(lookup, path, value):
    ctx = _context(
        "What time does it open?",
        {path: ResidentFact(value=value, source_turn_id="1", status="captured")},
    )

    output = await lookup({"near": "Flushing, Queens", "kind": "cooling_center"}, ctx)

    assert "NYC Cool Options" in output
    if path == "/cooling/site":
        assert path not in ctx.resident_facts
    else:
        assert ctx.resident_facts[path].value == {
            "keys": ["raices", "other"],
            "origin": [40.7580, -73.9780],
            "scope": {"kind": "cooling_center", "audience": "any"},
        }


@pytest.mark.asyncio
async def test_unknown_offered_scope_clears_selection_and_refreshes_results(lookup):
    ctx = _context(
        "What time does it open?",
        {
            "/cooling/offered": ResidentFact(
                value={
                    "keys": ["raices"],
                    "origin": [40.7580, -73.9780],
                    "scope": {"kind": "unknown", "audience": "any"},
                },
                source_turn_id="1",
                status="captured",
            ),
            "/cooling/site": ResidentFact(
                value={"key": "raices", "origin": [40.7580, -73.9780]},
                source_turn_id="1",
                status="captured",
            ),
        },
    )

    assert cooling._decode_site_fact(
        ctx.resident_facts["/cooling/offered"].value, offered=True
    ) is None

    output = await lookup({"near": "Flushing, Queens", "kind": "cooling_center"}, ctx)

    assert "Raices Times Square" in output
    assert "Closer Cooling Site" in output
    assert ctx.resident_facts["/cooling/offered"].value == {
        "keys": ["raices", "other"],
        "origin": [40.7580, -73.9780],
        "scope": {"kind": "cooling_center", "audience": "any"},
    }
    assert "/cooling/site" not in ctx.resident_facts


@pytest.mark.asyncio
async def test_negated_model_site_cannot_select_or_retain_prior_site(lookup):
    ctx = _context("Raices")
    await lookup({"near": "Flushing, Queens", "kind": "cooling_center"}, ctx)

    ctx.user_turns = ("Not Raices, show me other options",)
    output = await lookup(
        {
            "near": "Flushing, Queens",
            "kind": "cooling_center",
            "exclude_sites": ["Raices Times Square"],
        },
        ctx,
    )

    assert "Raices Times Square" not in output
    assert "Closer Cooling Site" in output
    assert "/cooling/site" not in ctx.resident_facts


@pytest.mark.asyncio
async def test_f182_lookup_returns_directions_from_the_resolved_origin(monkeypatch):
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
    assert "Step-free entrance: not confirmed by the City accessibility field" in output
    assert output.count("https://www.google.com/maps/dir/?api=1&origin=40.75800,-73.97800") == 2
    assert "destination=40.75920,-73.97610" in output
    assert "destination=40.75690,-73.96770" in output
    assert len(ctx.citations.mapping()) == 2
    assert all(
        citation["provenance"]["derivation"]["origin"] == [40.758, -73.978]
        for citation in ctx.citations.mapping().values()
    )
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
                "cc_wed_open1": "12:00 AM",
                "cc_wed_close1": "11:59 PM",
            }
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )
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
    assert "Other indoor Cool Options can be checked." in output
    assert "Try kind=" not in output


@pytest.mark.asyncio
async def test_current_cooling_lookup_fails_closed_when_all_rows_are_closed(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC_CLOSED",
                "Facility_name": "Flushing Library",
                "lat": 40.7581,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "05:00 PM",
            }
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 18, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = _context("Where can I cool down?")

    output = await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"}, ctx
    )

    assert output == (
        "No activated cooling center is confirmed open now. "
        "Other indoor Cool Options can be checked."
    )
    assert "Flushing Library" not in output


@pytest.mark.asyncio
async def test_current_cooling_lookup_fails_closed_when_hours_are_unknown(monkeypatch):
    handler = _patch_lookup(monkeypatch, [_site_row("unknown", "Unknown Hours Center")])

    output = await handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"},
        _context("Where can I cool down?"),
    )

    assert "No activated cooling center is confirmed open now" in output
    assert "Unknown Hours Center" not in output


@pytest.mark.asyncio
async def test_current_indoor_lookup_fails_closed_when_all_rows_are_closed(monkeypatch):
    row = {
        **_site_row("indoor-closed", "Flushing Library"),
        "Location_type": "Indoor",
        "Space_type": "Other Indoor Cool Option",
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "12:00 PM",
    }
    handler = _patch_lookup(monkeypatch, [row])

    output = await handler(
        {"near": "Flushing, Queens", "kind": "indoor"},
        _context("Where can I cool down indoors now?"),
    )

    assert "confirmed open now" in output
    assert "Flushing Library" not in output
    assert "123 W 42 ST" not in output
    assert "google.com/maps" not in output


@pytest.mark.asyncio
async def test_current_indoor_lookup_keeps_a_confirmed_open_row(monkeypatch):
    row = {
        **_site_row("indoor-open", "Open Indoor Option"),
        "Location_type": "Indoor",
        "Space_type": "Other Indoor Cool Option",
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "05:00 PM",
    }
    handler = _patch_lookup(monkeypatch, [row])

    output = await handler(
        {"near": "Flushing, Queens", "kind": "indoor"},
        _context("Where can I cool down indoors now?"),
    )

    assert "Open Indoor Option" in output
    assert "scheduled open now" in output
    result_line = next(line for line in output.splitlines() if line.startswith("1."))
    assert "rough estimate from the resolved place point, not a street address" in result_line


@pytest.mark.asyncio
async def test_cooling_lookup_keeps_precise_address_distance_unqualified(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(
            40.7580,
            -73.9780,
            "123 Main Street, Queens, NY",
            match_type="geosearch",
            bbl="4000000001",
        )

    row = {
        **_site_row("indoor-open", "Open Indoor Option"),
        "Location_type": "Indoor",
        "Space_type": "Other Indoor Cool Option",
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "05:00 PM",
    }
    handler = _patch_lookup(monkeypatch, [row])
    monkeypatch.setattr(cooling, "geocode", fake_geocode)

    output = await handler(
        {"near": "123 Main Street, Queens", "kind": "indoor"},
        _context("Where can I cool down indoors now?"),
    )

    result_line = next(line for line in output.splitlines() if line.startswith("1."))
    assert "rough estimate from the resolved place point" not in result_line


@pytest.mark.asyncio
async def test_current_all_lookup_does_not_bypass_closed_row_guard(monkeypatch):
    rows = [
        {
            **_site_row("indoor-closed", "Closed Indoor Option"),
            "Location_type": "Indoor",
            "Space_type": "Other Indoor Cool Option",
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "12:00 PM",
        },
        {
            **_site_row("center-closed", "Closed Cooling Center", 2),
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "12:00 PM",
        },
    ]
    handler = _patch_lookup(monkeypatch, rows)

    output = await handler(
        {"near": "Flushing, Queens", "kind": "all"},
        _context("Where can I cool down now?"),
    )

    assert "confirmed open now" in output
    assert "Closed Indoor Option" not in output
    assert "Closed Cooling Center" not in output
    assert "123 W 42 ST" not in output
    assert "google.com/maps" not in output


@pytest.mark.asyncio
async def test_current_cooling_lookup_keeps_a_confirmed_open_row(monkeypatch):
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.7580, -73.9780, text)

    async def fake_query(url, **kwargs):
        return [
            {
                "OBJECTID": 1,
                "NYCEM_ID": "CC_OPEN",
                "Facility_name": "Open Cooling Center",
                "lat": 40.7581,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "05:00 PM",
            },
            {
                "OBJECTID": 2,
                "NYCEM_ID": "CC_CLOSED",
                "Facility_name": "Closed Cooling Center",
                "lat": 40.7591,
                "lon": -73.9780,
                "Finder_status": "OPEN",
                "Space_type": "Cooling Center",
                "cc_wed_open1": "09:00 AM",
                "cc_wed_close1": "12:00 PM",
            },
        ]

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 7, 15, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    ctx = _context("Where can I cool down?")

    output = await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"}, ctx
    )

    assert "Open Cooling Center" in output
    assert "scheduled open now" in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_fields", [{}, {"cc_wed_open1": "09:00 AM", "cc_wed_close1": "12:00 PM"}]
)
async def test_selected_site_not_confirmed_open_does_not_replace_selection(
    monkeypatch, status_fields
):
    selected = {
        **_site_row("selected", "Selected Center"),
        **status_fields,
    }
    other = {
        **_site_row("other", "Other Open Center", 2, 40.7581),
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "05:00 PM",
    }
    handler = _patch_lookup(monkeypatch, [selected, other])
    facts = {
        "/cooling/offered": ResidentFact(
            value={
                "keys": ["selected", "other"],
                "origin": [40.7580, -73.9780],
                "scope": {"kind": "cooling_center", "audience": "any"},
            },
            source_turn_id="1",
            status="captured",
        ),
        "/cooling/site": ResidentFact(
            value={"key": "selected", "origin": [40.7580, -73.9780]},
            source_turn_id="1",
            status="captured",
        ),
    }

    output = await handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"},
        _context("Is it open now?", facts),
    )

    assert "selected cooling center is not confirmed open now" in output.lower()
    assert "other current cooling center options can be checked" in output.lower()
    assert "Other Open Center" not in output
    assert facts["/cooling/site"].value["key"] == "selected"


@pytest.mark.asyncio
async def test_negated_open_alternative_does_not_claim_current_options_exist(monkeypatch):
    selected = {
        **_site_row("selected", "Selected Center"),
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "12:00 PM",
    }
    other = {
        **_site_row("other", "Other Open Center", 2, 40.7581),
        "cc_wed_open1": "09:00 AM",
        "cc_wed_close1": "05:00 PM",
    }
    handler = _patch_lookup(monkeypatch, [selected, other])
    facts = {
        "/cooling/offered": ResidentFact(
            value={
                "keys": ["selected", "other"],
                "origin": [40.7580, -73.9780],
                "scope": {"kind": "cooling_center", "audience": "any"},
            },
            source_turn_id="1",
            status="captured",
        ),
        "/cooling/site": ResidentFact(
            value={"key": "selected", "origin": [40.7580, -73.9780]},
            source_turn_id="1",
            status="captured",
        ),
    }

    output = await handler(
        {
            "near": "Flushing, Queens",
            "kind": "cooling_center",
            "exclude_sites": ["Other Open Center"],
        },
        _context("Is Selected Center open now? Not Other Open Center.", facts),
    )

    assert "other current cooling center options can be checked" not in output.lower()
    assert "other indoor cool options can be checked" in output.lower()
    assert facts["/cooling/site"].value["key"] == "selected"


@pytest.mark.asyncio
async def test_selected_site_unknown_and_no_open_alternative_fails_closed(monkeypatch):
    handler = _patch_lookup(monkeypatch, [_site_row("selected", "Selected Center")])
    facts = {
        "/cooling/offered": ResidentFact(
            value={
                "keys": ["selected"],
                "origin": [40.7580, -73.9780],
                "scope": {"kind": "cooling_center", "audience": "any"},
            },
            source_turn_id="1",
            status="captured",
        ),
        "/cooling/site": ResidentFact(
            value={"key": "selected", "origin": [40.7580, -73.9780]},
            source_turn_id="1",
            status="captured",
        ),
    }

    output = await handler(
        {"near": "Flushing, Queens", "kind": "cooling_center"},
        _context("Is it open now?", facts),
    )

    assert "selected cooling center is not confirmed open now" in output.lower()
    assert "other indoor cool options can be checked" in output.lower()
    assert "Selected Center" not in output
    assert facts["/cooling/site"].value["key"] == "selected"


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
async def test_closer_closed_summary_uses_nearest_open_not_feed_order(monkeypatch):
    rows = [
        {
            **_site_row("far", "Far Open Center", lat=40.7870),
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "05:00 PM",
        },
        {
            **_site_row("closed", "Closed Center", 2, 40.7600),
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "12:00 PM",
            "cc_thu_open1": "09:30 AM",
            "cc_thu_close1": "05:00 PM",
        },
        {
            **_site_row("nearest", "Nearest Open Center", 3, 40.7585),
            "cc_wed_open1": "09:00 AM",
            "cc_wed_close1": "05:00 PM",
        },
    ]
    lookup = _patch_lookup(monkeypatch, rows)
    ctx = _context("Where can I cool down?")

    output = await lookup({"near": "Times Square", "kind": "cooling_center"}, ctx)

    assert "Nearest Open Center" in output
    assert "closer option" not in output
    assert "soonest reopens" not in output


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
