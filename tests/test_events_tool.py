from __future__ import annotations

import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.modules.events import tools as events
from heynyc.modules.events.tools import (
    Event,
    _event_block,
    _explicitly_free,
    _from_parks,
    _from_ticketmaster,
    _future_only,
    _requested_window,
    get_tools,
)


def test_from_ticketmaster_maps_fields():
    raw = {
        "name": "FIFA Final Watch Party",
        "url": "https://www.ticketmaster.com/event/abc",
        "dates": {"start": {"localDate": "2026-07-19", "localTime": "15:00:00"}},
        "_embedded": {"venues": [{"name": "Central Park", "city": {"name": "New York"}}]},
    }
    ev = _from_ticketmaster(raw)
    assert ev == Event(
        name="FIFA Final Watch Party", start_date="2026-07-19", start_time="15:00:00",
        venue="Central Park", borough="New York",
        url="https://www.ticketmaster.com/event/abc", source="Ticketmaster", tier="authoritative",
    )


def test_from_ticketmaster_drops_a_metro_event_outside_nyc():
    raw = {
        "name": "Museum at Bethel Woods",
        "url": "https://www.ticketmaster.com/event/bethel",
        "dates": {"start": {"localDate": "2026-08-12", "localTime": "10:00:00"}},
        "_embedded": {"venues": [{"name": "Bethel Woods", "city": {"name": "Bethel"}}]},
    }

    assert _from_ticketmaster(raw) is None


def test_from_ticketmaster_drops_dateless():
    assert _from_ticketmaster({"name": "TBA", "dates": {"start": {}}}) is None
    assert _from_ticketmaster({"name": "Bad", "dates": {"start": {"localDate": "later"}}}) is None


def test_from_ticketmaster_drops_cancelled_or_postponed_events():
    for status in ("canceled", "cancelled", "postponed"):
        raw = {
            "name": "No longer happening",
            "dates": {"start": {"localDate": "2026-07-19"}, "status": {"code": status}},
        }
        assert _from_ticketmaster(raw) is None


def test_from_ticketmaster_normalizes_null_name():
    event = _from_ticketmaster({
        "name": None, "dates": {"start": {"localDate": "2026-07-19"}},
    })
    assert event is not None
    assert event.name == ""


def test_from_parks_maps_nested_link():
    raw = {
        "title": "Summer Pickleball",
        "startdate": "2026-06-17T00:00:00.000",
        "starttime": "7:00 am",
        "parknames": "Blood Root Valley",
        "location": "Arts and Crafts Room",
        "link": {"url": "http://www.nycgovparks.org/events/2026/06/17/x"},
    }
    ev = _from_parks(raw)
    assert ev.name == "Summer Pickleball"
    assert ev.start_date == "2026-06-17"
    assert ev.start_time == "7:00 am"
    assert ev.venue == "Blood Root Valley"
    assert ev.url == "https://www.nycgovparks.org/events/2026/06/17/x"
    assert ev.source == "NYC Parks" and ev.tier == "authoritative"


def test_from_parks_preserves_source_free_audience_and_borough_fields():
    ev = _from_parks({
        "title": "NYRR Open Run: Cunningham Park",
        "description": "The program is free and open to runners and walkers of all ages.",
        "categories": "Best for Kids | Running/Jogging",
        "parkids": "Q021",
        "parknames": "Cunningham Park",
        "startdate": "2026-07-26T00:00:00.000",
        "starttime": "2026-07-20 15:00:00",
    })

    assert ev is not None
    assert ev.borough == "Queens"
    assert ev.start_time == "3:00 PM"
    assert "free" in ev.free_evidence.lower()
    assert ev.audience == "Best for Kids"
    assert _explicitly_free([ev], "free events in Queens") == [ev]
    block = _event_block(ev, "S1")
    assert "free" in block.lower()
    assert "Best for Kids" in block


def test_from_parks_drops_cancelled_titles():
    for title in ("CANCELLED: Movie Night", "Canceled - Outdoor Concert", "POSTPONED: Movie"):
        assert _from_parks({"title": title, "startdate": "2026-07-19"}) is None


def test_from_parks_handles_null_title():
    event = _from_parks({"title": None, "startdate": "2026-07-19"})
    assert event is not None
    assert event.name == ""


def test_from_parks_drops_malformed_date():
    assert _from_parks({"title": "Bad upstream row", "startdate": "not-a-date"}) is None


def test_from_permitted_maps_sapo_fields():
    raw = {
        "event_id": "914332",
        "event_name": "HHFM Jacobi Hospital Market",
        "start_date_time": "2026-07-18T08:00:00.000",
        "end_date_time": "2026-07-18T15:00:00.000",
        "event_agency": "Street Activity Permit Office",
        "event_type": "Farmers Market",
        "event_borough": "Bronx",
        "event_location": "PELHAM PARKWAY SOUTH between WILSON AVENUE and EASTCHESTER ROAD",
        ":id": "row-7m8x~wksu-hgwb",
    }
    ev = events._from_permitted(raw)
    assert ev.name == "HHFM Jacobi Hospital Market"
    assert ev.start_date == "2026-07-18"
    assert ev.start_time == "08:00"  # HH:MM lifted from the ISO start_date_time
    assert ev.end_time == "15:00"
    assert ev.venue == "PELHAM PARKWAY SOUTH between WILSON AVENUE and EASTCHESTER ROAD"
    assert ev.borough == "Bronx"
    assert ev.source == "NYC Permitted Events" and ev.tier == "authoritative"
    # DATA citation points at the re-fetchable Socrata row permalink for this dataset.
    assert "tvpp-9vvx" in ev.url and "row-7m8x~wksu-hgwb" in ev.url


def test_from_permitted_drops_dateless_and_mirrors_cancellation_discipline():
    assert events._from_permitted({"event_name": "No date"}) is None
    assert events._from_permitted({"event_name": "Bad", "start_date_time": "nope"}) is None
    # The permit dataset carries no status column, so mirror the Parks/Ticketmaster discipline:
    # a permit whose name marks it cancelled/postponed is not recommended as happening.
    for name in ("CANCELLED: Block Party", "Canceled Street Fair", "POSTPONED Parade"):
        assert events._from_permitted({
            "event_name": name, "start_date_time": "2026-07-19T10:00:00.000",
        }) is None


def test_from_permitted_falls_back_to_dataset_page_without_row_id():
    ev = events._from_permitted({
        "event_name": None, "start_date_time": "2026-07-19T10:00:00.000",
    })
    assert ev is not None
    assert ev.name == ""
    assert ev.url == events.PERMITTED_SOURCE_URL


def test_future_only_filters_past():
    past = Event("old", "2026-06-01", "", "", "", "u", "NYC Parks", "authoritative")
    future = Event("new", "2026-07-19", "", "", "", "u", "NYC Parks", "authoritative")
    kept = _future_only([past, future], today="2026-06-28")
    assert kept == [future]


def test_known_finished_events_do_not_block_current_results():
    now = events.datetime(2026, 7, 18, 23, 0, tzinfo=events.NYC_TZ)
    finished = Event(
        "Morning market", "2026-07-18", "08:00", "Plaza", "Queens", "u1",
        "NYC Permitted Events", "authoritative", end_time="15:00",
    )
    still_open = Event(
        "Night market", "2026-07-18", "18:00", "Plaza", "Queens", "u2",
        "NYC Permitted Events", "authoritative", end_time="23:30",
    )
    unknown_end = Event(
        "Concert", "2026-07-18", "20:00", "Park", "Queens", "u3",
        "Ticketmaster", "authoritative",
    )
    unknown_start = Event(
        "Date-only listing", "2026-07-18", "", "Arena", "Queens", "u4",
        "Ticketmaster", "authoritative",
    )
    later = Event(
        "Late show", "2026-07-18", "23:30", "Arena", "Queens", "u5",
        "Ticketmaster", "authoritative",
    )

    assert events._not_ended_today(
        [finished, still_open, unknown_end, unknown_start, later], now,
    ) == [still_open, later]


def test_requested_window_resolves_this_weekend_from_nyc_date():
    assert _requested_window("free events this weekend", "2026-07-16") == (
        "2026-07-18", "2026-07-19",
    )
    assert _requested_window("events this weekend", "2026-07-18") == (
        "2026-07-18", "2026-07-19",
    )
    assert _requested_window("events today", "2026-07-16") == (
        "2026-07-16", "2026-07-16",
    )
    assert _requested_window("what is happening tonight", "2026-07-16") == (
        "2026-07-16", "2026-07-16",
    )
    assert _requested_window("things to do this week", "2026-07-16") == (
        "2026-07-16", "2026-07-19",
    )


def test_requested_window_resolves_tomorrow():
    assert _requested_window("what to prepare for tomorrows wc game", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    assert _requested_window("what should i bring to the game tmrw", "2026-07-31") == (
        "2026-08-01", "2026-08-01",
    )
    assert _requested_window("what game is happening tomorow", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    assert _requested_window("events at the department", "2026-07-17") == ("2026-07-17", None)


def test_requested_window_resolves_numeric_dates():
    assert _requested_window("what should i bring to the game on 7/18", "2026-07-17") == (
        "2026-07-18", "2026-07-18",
    )
    # A month/day earlier in the year means the next occurrence.
    assert _requested_window("the parade on 1/1", "2026-07-17") == ("2027-01-01", "2027-01-01")
    # Invalid calendar dates fall through to the default window.
    assert _requested_window("ratio is 19/32 exactly", "2026-07-17") == ("2026-07-17", None)


def test_free_filter_requires_source_title_and_event_block_supplies_weekday():
    listed_free = Event(
        "Free Yoga", "2026-07-18", "9:00 AM", "Park", "", "u1", "NYC Parks", "authoritative",
    )
    unknown_cost = Event(
        "Open Run", "2026-07-18", "9:00 AM", "Park", "", "u2", "NYC Parks", "authoritative",
    )

    assert _explicitly_free([listed_free, unknown_cost], "free events") == [listed_free]
    assert _explicitly_free([listed_free, unknown_cost], "events") == [listed_free, unknown_cost]
    assert "Saturday, 2026-07-18" in _event_block(listed_free, "S1")
    assert "Details: u1" in _event_block(listed_free, "S1")


def test_free_is_a_cost_filter_not_an_event_keyword():
    free_yoga = Event(
        "Free Yoga", "2026-07-18", "9:00 AM", "Park", "Queens", "u1",
        "NYC Parks", "authoritative", free_evidence="This program is free",
    )
    free_movie = Event(
        "Movies Under the Stars", "2026-07-18", "8:00 PM", "Park", "Queens", "u2",
        "NYC Parks", "authoritative", free_evidence="This program is free",
    )

    assert not events._matches_keyword(free_yoga, "free music or outdoor movies")
    assert events._matches_keyword(free_movie, "free music or outdoor movies")


async def test_broad_event_intent_is_not_sent_as_a_catalog_keyword(monkeypatch):
    keywords_seen = []

    async def fake_ticketmaster(**kwargs):
        keywords_seen.append(("ticketmaster", kwargs.get("keyword")))
        return []

    async def fake_query(dataset_id, **kwargs):
        keywords_seen.append((dataset_id, kwargs.get("q")))
        if dataset_id == events.PARKS_DATASET_ID:
            return [{
                "title": "Evening Concert",
                "startdate": "2099-08-11",
                "starttime": "7:00 PM",
                "parknames": "Central Park",
                "link": {"url": "https://www.nycgovparks.org/events/evening-concert"},
            }]
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="what to do in nyc today",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "things to do",
            "window_start": "2099-08-11",
            "window_end": "2099-08-11",
        },
        ctx,
    )

    assert set(keywords_seen) == {
        ("ticketmaster", None),
        (events.PARKS_DATASET_ID, None),
        (events.PERMITTED_DATASET_ID, None),
    }
    assert "Evening Concert" in output
    assert "BROADENED" not in output


async def test_broad_event_lookup_runs_one_untimed_web_lane_in_parallel(monkeypatch):
    calls = []

    async def no_ticketmaster(**kwargs):
        return []

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_handler(args, ctx):
        calls.append(args)
        return "[S1] Current guide\nThree things happening today"

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", no_city_rows)
    web_tool = Tool(
        name="web_search",
        description="search",
        parameters={"type": "object"},
        handler=web_handler,
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="what to do in nyc today",
        toolbox={"web_search": web_tool},
    )

    output = await get_tools()[0].handler({}, ctx)

    assert calls == [{"query": "what to do in nyc today", "count": 5}]
    assert "Current web event leads" in output
    assert "Three things happening today" in output


async def test_constrained_event_lookup_uses_one_model_shaped_web_lane(monkeypatch):
    calls = []

    async def no_ticketmaster(**kwargs):
        return []

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_handler(args, ctx):
        calls.append(args)
        return "[S1] Official constrained event lead"

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="what free things can I do with a toddler in Flushing this Saturday if it rains?",
        toolbox={
            "web_search": Tool(
                name="web_search",
                description="search",
                parameters={"type": "object"},
                handler=web_handler,
            )
        },
    )

    output = await get_tools()[0].handler(
        {
            "audience": "kids",
            "setting": "indoor",
            "borough": "Queens",
            "window_start": "2099-08-15",
            "window_end": "2099-08-15",
            "web_query": "free indoor toddler events Flushing 2099-08-15",
        },
        ctx,
    )

    assert calls == [{"query": "free indoor toddler events Flushing 2099-08-15", "count": 5}]
    assert "Official constrained event lead" in output
    assert "Requested setting: indoor" in output
    assert "do not infer indoor or outdoor" in output


async def test_broad_event_lookup_discloses_an_unavailable_web_lane(monkeypatch):
    async def no_ticketmaster(**kwargs):
        return []

    async def city_rows(dataset_id, **kwargs):
        if dataset_id == events.PARKS_DATASET_ID:
            return [{
                "title": "Evening Concert",
                "startdate": "2099-08-12",
                "starttime": "7:00 PM",
                "parknames": "Central Park",
                "link": {"url": "https://www.nycgovparks.org/events/evening-concert"},
            }]
        return []

    async def broken_web(_args, _ctx):
        raise httpx.ReadTimeout("search unavailable")

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="what to do in nyc on August 12, 2099",
        toolbox={
            "web_search": Tool(
                name="web_search",
                description="search",
                parameters={"type": "object"},
                handler=broken_web,
            )
        },
    )

    output = await get_tools()[0].handler(
        {"window_start": "2099-08-12", "window_end": "2099-08-12"}, ctx
    )

    assert "Evening Concert" in output
    assert "Current web event leads were unavailable" in output
    assert "Results are partial" in output
    assert "do not claim complete coverage" not in output


async def test_specific_event_lookup_leaves_web_search_for_model_followup(monkeypatch):
    calls = []

    async def no_ticketmaster(**kwargs):
        return []

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_handler(args, ctx):
        calls.append(args)
        return "unexpected"

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Knicks game tonight",
        toolbox={
            "web_search": Tool(
                name="web_search",
                description="search",
                parameters={"type": "object"},
                handler=web_handler,
            )
        },
    )

    await get_tools()[0].handler({"keyword": "Knicks"}, ctx)

    assert calls == []


def test_event_block_flags_a_today_event_whose_start_time_already_passed():
    """F065: a today-dated event whose local start time is already past `now` gets a deterministic,
    language-independent 'already started or ended' note in the tool line, so a finished event is
    not offered as still attendable. Data-shaped: only known start time vs now, no model text."""
    now = events.datetime(2026, 7, 20, 15, 0, tzinfo=events.NYC_TZ)
    started = Event("Noon Rally", "2026-07-20", "12:00 PM", "City Hall", "", "u", "NYC Parks", "authoritative")
    upcoming = Event("Evening Show", "2026-07-20", "8:00 PM", "SummerStage", "", "u", "NYC Parks", "authoritative")
    tomorrow = Event("Tomorrow Fair", "2026-07-21", "9:00 AM", "Park", "", "u", "NYC Parks", "authoritative")
    undated = Event("No Time", "2026-07-20", "", "Park", "", "u", "NYC Parks", "authoritative")

    assert "already started or ended" in events._event_block(started, "S1", now)
    assert "already started or ended" not in events._event_block(upcoming, "S2", now)
    assert "already started or ended" not in events._event_block(tomorrow, "S3", now)
    assert "already started or ended" not in events._event_block(undated, "S4", now)
    assert "starts later today" in events._event_block(upcoming, "S2", now)
    assert "starts later today" not in events._event_block(started, "S1", now)
    assert "starts later today" not in events._event_block(tomorrow, "S3", now)
    assert "starts later today" not in events._event_block(undated, "S4", now)
    in_progress = Event(
        "Greenmarket", "2026-07-20", "8:00 AM", "Madison Avenue", "Manhattan",
        "u", "NYC Permitted Events", "authoritative", end_time="6:00 PM",
    )
    assert "in progress" in events._event_block(in_progress, "S5", now)
    assert "ends 6:00 PM" in events._event_block(in_progress, "S5", now)
    # Back-compat: without a `now` reference there is no annotation (existing callers/tests).
    assert "already started or ended" not in events._event_block(started, "S1")
    assert "starts later today" not in events._event_block(upcoming, "S2")


async def test_event_citation_contains_end_time_and_derived_timing_status(monkeypatch):
    class FixedDateTime(events.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 12, 12, 0, tzinfo=tz)

    async def no_ticketmaster(**kwargs):
        return []

    async def city_rows(dataset_id, **kwargs):
        if dataset_id != events.PERMITTED_DATASET_ID:
            return []
        return [{
            "event_name": "Noon Plaza Event",
            "start_date_time": "2026-08-12T10:00:00.000",
            "end_date_time": "2026-08-12T14:00:00.000",
            "event_borough": "Manhattan",
            "event_location": "City Hall Park",
            ":id": "event-row",
        }]

    monkeypatch.setattr(events, "datetime", FixedDateTime)
    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="plaza events today",
    )

    output = await get_tools()[0].handler({}, ctx)
    citation = next(iter(ctx.citations.mapping().values()))

    assert "in progress" in output
    assert "14:00" in citation["snippet"]
    assert "in progress" in citation["snippet"]


def test_shortlist_deduplicates_ticket_products_for_one_venue_and_time():
    rows = [
        Event(
            "Banksy Museum - Flexiticket", "2026-08-12", "10:00:00",
            "Banksy Museum New York", "New York", "u1", "Ticketmaster", "authoritative",
        ),
        Event(
            "The Banksy Museum New York!", "2026-08-12", "10:00:00",
            "Banksy Museum New York", "New York", "u2", "Ticketmaster", "authoritative",
        ),
    ]

    assert events._shortlist(rows, 12) == [rows[0]]


def test_tonight_filter_keeps_only_parseable_future_evening_events():
    morning = Event("Morning", "2026-07-16", "9:00 am", "", "", "u1", "NYC Parks", "authoritative")
    evening = Event("Evening", "2026-07-16", "7:00 pm", "", "", "u2", "NYC Parks", "authoritative")
    unknown = Event("Unknown", "2026-07-16", "", "", "", "u3", "NYC Parks", "authoritative")
    now = events.datetime(2026, 7, 16, 15, 0, tzinfo=events.NYC_TZ)

    assert events._tonight_only([morning, evening, unknown], now) == [evening]


@pytest.fixture(autouse=True)
def _force_tm_key(monkeypatch):
    # Force the TM branch to run offline (the handler reads config.TICKETMASTER_API_KEY).
    monkeypatch.setattr("heynyc.core.ticketmaster.config.TICKETMASTER_API_KEY", "test-key")


def _routed_client() -> httpx.AsyncClient:
    """One client routing Ticketmaster vs Socrata by host — both fully offline."""
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "ticketmaster" in host:
            return httpx.Response(200, json={"_embedded": {"events": [{
                "name": "Concert in the Park",
                "url": "https://www.ticketmaster.com/event/abc",
                "dates": {"start": {"localDate": "2099-07-19", "localTime": "20:00:00"}},
                "_embedded": {"venues": [{"name": "SummerStage", "city": {"name": "New York"}}]},
            }]}})
        # Socrata (Parks): one past row (filtered out) + one future row.
        return httpx.Response(200, json=[
            {"title": "Old Festival", "startdate": "2000-01-01T00:00:00.000",
             "starttime": "9:00 am", "parknames": "Old Park",
             "link": {"url": "http://www.nycgovparks.org/events/old"}},
            {"title": "Future Fair", "startdate": "2099-08-01T00:00:00.000",
             "starttime": "10:00 am", "parknames": "New Park",
             "link": {"url": "http://www.nycgovparks.org/events/new"}},
        ])
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_find_nyc_events_merges_grounds_and_filters_future():
    [tool] = get_tools()
    citations = CitationRegistry()
    async with _routed_client() as client:
        ctx = ToolContext(citations=citations, registry=Registry([]), http=client)
        out = await tool.handler({}, ctx)

    assert "Concert in the Park" in out
    assert "Future Fair" in out
    assert "Old Festival" not in out          # past event filtered (§12)
    assert "{cite:" in out                     # everything is grounded + cited
    assert citations.mapping()                 # at least one DATA citation registered


async def test_find_nyc_events_grounds_the_official_indexes_when_no_event_matches(
    monkeypatch,
):
    async def no_ticketmaster(**kwargs):
        return []

    async def no_city_rows(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", no_city_rows)
    citations = CitationRegistry()
    ctx = ToolContext(
        citations=citations,
        registry=Registry([]),
        query="where could i watch the 2030 final in nyc",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "2030 World Cup watch parties",
            "classification": "Sports",
            "borough": "Queens",
            "window_start": "2030-07-20",
            "window_end": "2030-07-21",
        },
        ctx,
    )

    assert f"NYC Parks {events.PARKS_SOURCE_URL}" in output
    assert f"NYC Permitted Events {events.PERMITTED_SOURCE_URL}" in output
    assert "Say that this lookup could not confirm a match" not in output
    assert "do not claim that no matching events exist" not in output
    assert output.count("{cite:") == 2
    assert {citation["url"] for citation in citations.mapping().values()} == {
        events.PARKS_SOURCE_URL,
        events.PERMITTED_SOURCE_URL,
    }
    assert {citation["kind"] for citation in citations.mapping().values()} == {"WEB"}
    assert "Use the resident's own time wording" not in output


def test_find_nyc_events_borough_schema_keeps_citywide_requests_citywide():
    [tool] = get_tools()

    assert "only when the resident names one" in (
        tool.parameters["properties"]["borough"]["description"]
    )
    assert "NYC means citywide" in (
        tool.parameters["properties"]["borough"]["description"]
    )


async def test_find_nyc_events_retries_only_failed_catalog_sources(monkeypatch):
    attempts = 0

    async def flaky_ticketmaster(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("transient")
        return [{
            "name": "Mets vs. Braves",
            "url": "https://www.ticketmaster.com/event/game",
            "dates": {"start": {"localDate": "2099-07-19", "localTime": "19:10:00"}},
        }]

    async def other_sources(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", flaky_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", other_sources)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="what game happened today",
    )

    output = await get_tools()[0].handler(
        {"window_start": "2099-07-19", "window_end": "2099-07-19"}, ctx,
    )

    assert attempts == 2
    assert "Mets vs. Braves" in output
    assert "Results are partial" not in output


async def test_find_nyc_events_discloses_a_catalog_source_that_stays_unavailable(monkeypatch):
    attempts = 0

    async def broken_ticketmaster(**kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("still unavailable")

    async def other_sources(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", broken_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", other_sources)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="what game happened today",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "game",
            "window_start": "2099-07-19",
            "window_end": "2099-07-19",
        },
        ctx,
    )

    assert attempts == 2
    assert "Sources unavailable for part of this lookup: Ticketmaster" in output
    assert "Results are partial" in output
    assert "do not claim complete coverage or that no matching event exists" not in output
    assert "No matching events were confirmed from the sources that responded" in output
    assert "No upcoming NYC events matched" not in output


async def test_find_nyc_events_includes_permitted_street_events():
    """The permitted-events lane surfaces a Street Activity Permit Office row (street fair /
    farmers market) our Ticketmaster + Parks lanes structurally miss, cited to its Socrata row."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "ticketmaster" in request.url.host:
            return httpx.Response(200, json={"_embedded": {"events": []}})
        if events.PERMITTED_DATASET_ID in request.url.path:
            return httpx.Response(200, json=[{
                "event_id": "931866",
                "event_name": "Inwood Greenmarket",
                "start_date_time": "2099-07-18T08:00:00.000",
                "end_date_time": "2099-07-18T15:00:00.000",
                "event_agency": "Street Activity Permit Office",
                "event_type": "Farmers Market",
                "event_borough": "Manhattan",
                "event_location": "ISHAM STREET between COOPER STREET and SEAMAN AVENUE",
                ":id": "row-abcd-1234",
            }])
        return httpx.Response(200, json=[])  # Parks: empty

    citations = CitationRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ctx = ToolContext(
            citations=citations, registry=Registry([]), http=client, query="any street fairs",
        )
        out = await get_tools()[0].handler({}, ctx)

    assert "Inwood Greenmarket" in out
    assert "NYC Permitted Events" in out
    assert any("tvpp-9vvx" in c["url"] for c in citations.mapping().values())


async def test_find_nyc_events_routes_to_indexes_when_only_permits_already_ended(
    monkeypatch,
):
    class FixedDateTime(events.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 18, 23, 0, tzinfo=tz)

    async def no_ticketmaster(**kwargs):
        return []

    async def city_rows(dataset_id, **kwargs):
        if dataset_id != events.PERMITTED_DATASET_ID:
            return []
        return [{
            "event_name": "Elmhurst Greenmarket",
            "start_date_time": "2026-07-18T08:00:00.000",
            "end_date_time": "2026-07-18T15:00:00.000",
            "event_borough": "Queens",
            "event_location": "Elmhurst",
        }]

    monkeypatch.setattr(events, "datetime", FixedDateTime)
    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="What free events are still happening in Queens today?",
    )

    output = await get_tools()[0].handler({"borough": "Queens"}, ctx)

    assert "Elmhurst Greenmarket" not in output
    assert events.PARKS_SOURCE_URL in output
    assert events.PERMITTED_SOURCE_URL in output


async def test_find_nyc_events_never_falls_back_across_requested_borough(
    monkeypatch,
):
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_query(dataset_id, **kwargs):
        if dataset_id != events.PERMITTED_DATASET_ID:
            return []
        return [{
            "event_name": "Bronx Event",
            "start_date_time": "2099-07-25T10:00:00.000",
            "event_agency": "Street Activity Permit Office",
            "event_type": "Block Party",
            "event_borough": "Bronx",
            "event_location": "Grand Concourse",
        }]

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="events",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "kids",
            "borough": "Queens",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert output.startswith(events._NO_RESULTS)
    assert "Bronx Event" not in output


async def test_find_nyc_events_keeps_a_matching_requested_borough(monkeypatch):
    web_calls = 0

    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_query(dataset_id, **kwargs):
        if dataset_id != events.PERMITTED_DATASET_ID:
            return []
        return [{
            "event_name": "Queens Event",
            "start_date_time": "2099-07-25T10:00:00.000",
            "event_agency": "Street Activity Permit Office",
            "event_type": "Block Party",
            "event_borough": "Queens",
            "event_location": "Flushing",
        }]

    async def web_handler(args, ctx):
        nonlocal web_calls
        web_calls += 1
        return "Bronx Event from web search"

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="events on Saturday",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "kids",
            "borough": "Queens",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert "Queens Event" in output
    assert "Bronx Event" not in output
    assert web_calls == 0


async def test_find_nyc_events_keeps_free_parks_rows_in_the_requested_borough(monkeypatch):
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_query(dataset_id, **kwargs):
        if dataset_id != events.PARKS_DATASET_ID:
            return []
        return [
            {
                "title": "Queens Open Run",
                "description": "This program is free and open to all ages.",
                "categories": "Best for Kids | Running/Jogging",
                "parkids": "Q021",
                "parknames": "Cunningham Park",
                "startdate": "2099-07-25T00:00:00.000",
            },
            {
                "title": "Brooklyn Open Run",
                "description": "This program is free and open to all ages.",
                "parkids": "B057",
                "parknames": "Marine Park",
                "startdate": "2099-07-25T00:00:00.000",
            },
            {
                "title": "Queens General Concert",
                "description": "This concert is free.",
                "categories": "Concerts",
                "parkids": "Q021",
                "parknames": "Cunningham Park",
                "startdate": "2099-07-25T00:00:00.000",
            },
        ]

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]),
        query="free events for kids in Queens",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "free events",
            "borough": "Queens",
            "audience": "kids",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert "Queens Open Run" in output
    assert "Best for Kids" in output
    assert "Queens General Concert" not in output
    assert "Brooklyn Open Run" not in output
    assert get_tools()[0].parameters["properties"]["audience"]["enum"] == ["kids"]


async def test_find_nyc_events_does_not_mix_unfiltered_context_into_audience_results(
    monkeypatch,
):
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_query(dataset_id, **kwargs):
        if dataset_id != events.PARKS_DATASET_ID:
            return []
        return [
            {
                "title": "Kids In Motion",
                "description": "This program is free.",
                "categories": "Best for Kids",
                "parkids": "Q021",
                "startdate": "2099-07-25T00:00:00.000",
            },
            {
                "title": "General Concert",
                "description": "This concert is free.",
                "categories": "Concerts",
                "parkids": "Q021",
                "startdate": "2099-07-25T00:00:00.000",
            },
        ]

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="free kids events citywide",
        event_turn="discovery",
    )

    with pytest.raises(ValueError, match="Unsupported audience"):
        await get_tools()[0].handler({"audience": "families"}, ctx)

    output = await get_tools()[0].handler(
        {
            "audience": "kids",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert "Kids In Motion" in output
    assert "General Concert" not in output
    assert "Unfiltered adult event" not in output


async def test_find_nyc_events_suppresses_unverified_editorial_boroughs(monkeypatch):
    editorial_calls = 0

    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_query(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="events on Saturday",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler(
        {
            "borough": "Queens",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert output.startswith(events._NO_RESULTS)
    assert output.count("{cite:") == 2
    assert events.PARKS_SOURCE_URL in output
    assert events.PERMITTED_SOURCE_URL in output
    assert "Bronx Event" not in output
    assert editorial_calls == 0


async def test_permitted_lane_filters_sport_noise_by_agency_field_not_event_name(monkeypatch):
    """Hard rule: cut the ~6.5k/week sport-reservation noise by the dataset's own agency/type
    fields, never by keyword-matching event names."""
    captured: dict[str, object] = {}

    async def fake_query(dataset_id, **kwargs):
        if dataset_id == events.PERMITTED_DATASET_ID:
            captured["where"] = kwargs.get("where")
        return []

    async def fake_tm(**kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_tm)
    monkeypatch.setattr(events, "query_dataset", fake_query)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="any street fairs",
    )

    await get_tools()[0].handler({}, ctx)

    where = captured["where"]
    assert "event_agency" in where and "Street Activity Permit Office" in where
    assert "Production Event" in where          # exclude the production/load-in noise by type
    assert "event_name" not in where            # never keyword-match event names


async def test_broad_weekend_query_seats_permitted_alongside_ticketmaster_and_parks(monkeypatch):
    """Live gap: a broad no-keyword weekend query collapses ~70 Ticketmaster+Parks rows onto the
    weekend's one or two dates, so a plain date sort (stable, last-appended lane loses) truncates
    the permitted lane to zero before the cap. The merged shortlist must seat at least one
    permitted street-event row alongside the Ticketmaster and Parks rows."""
    today = events.datetime.now(events.NYC_TZ).strftime("%Y-%m-%d")
    window_start, _window_end = events._requested_window("what's happening this weekend", today)

    def handler(request: httpx.Request) -> httpx.Response:
        if "ticketmaster" in request.url.host:
            # Enough same-date rows to saturate the default cap of 12 on their own.
            return httpx.Response(200, json={"_embedded": {"events": [
                {
                    "name": f"Ticketmaster Show {i}",
                    "url": f"https://www.ticketmaster.com/event/{i}",
                    "dates": {"start": {"localDate": window_start, "localTime": "20:00:00"}},
                    "_embedded": {"venues": [{"name": "Arena", "city": {"name": "New York"}}]},
                }
                for i in range(12)
            ]}})
        if events.PERMITTED_DATASET_ID in request.url.path:
            return httpx.Response(200, json=[{
                "event_id": "931866",
                "event_name": "Inwood Greenmarket",
                "start_date_time": f"{window_start}T08:00:00.000",
                "event_agency": "Street Activity Permit Office",
                "event_type": "Farmers Market",
                "event_borough": "Manhattan",
                "event_location": "ISHAM STREET between COOPER STREET and SEAMAN AVENUE",
                ":id": "row-abcd-1234",
            }])
        # Parks: two same-date rows.
        return httpx.Response(200, json=[
            {"title": "Parks Concert", "startdate": f"{window_start}T00:00:00.000",
             "starttime": "10:00 am", "parknames": "Central Park",
             "link": {"url": "http://www.nycgovparks.org/events/a"}},
            {"title": "Parks Movie Night", "startdate": f"{window_start}T00:00:00.000",
             "starttime": "8:00 pm", "parknames": "Prospect Park",
             "link": {"url": "http://www.nycgovparks.org/events/b"}},
        ])


    citations = CitationRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ctx = ToolContext(
            citations=citations, registry=Registry([]), http=client,
            query="what's happening this weekend",
        )
        out = await get_tools()[0].handler({}, ctx)

    assert "Inwood Greenmarket" in out          # permitted row competes despite the cap
    assert "Ticketmaster Show" in out            # alongside Ticketmaster
    assert "Parks" in out                        # and Parks
    assert any("tvpp-9vvx" in c["url"] for c in citations.mapping().values())


async def test_keyword_broadening_does_not_return_unrelated_events(monkeypatch):
    async def no_ticketmaster(**kwargs):
        return []

    async def unrelated_when_unkeyworded(dataset_id, **kwargs):
        if dataset_id == events.PARKS_DATASET_ID:
            return [{
                "title": "Central Park Movies Under the Stars",
                "description": "A free outdoor film series.",
                "startdate": "2099-07-25T18:00:00.000",
                "parknames": "Central Park",
                "parkids": "M010",
            }]
        if kwargs.get("q"):
            return []
        if dataset_id == events.PERMITTED_DATASET_ID:
            return [{
                "event_name": "Generic Fitness Class",
                "start_date_time": "2099-07-25T10:00:00.000",
                "event_agency": "Street Activity Permit Office",
                "event_type": "Plaza Event",
                "event_borough": "Manhattan",
                "event_location": "Herald Square",
            }]
        return []

    monkeypatch.setattr(events, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", unrelated_when_unkeyworded)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="where can I watch the world cup in NYC?",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler(
        {
            "keyword": "world cup",
            "borough": "Manhattan",
            "window_start": "2099-07-25",
            "window_end": "2099-07-25",
        },
        ctx,
    )

    assert output.startswith(events._NO_RESULTS)
    assert "Generic Fitness Class" not in output
    assert {
        citation["url"] for citation in ctx.citations.mapping().values()
    } == {events.PARKS_SOURCE_URL, events.PERMITTED_SOURCE_URL}


def test_find_nyc_events_contract_explains_broad_and_specific_search_ownership():
    from heynyc.modules.events.tools import get_tools

    desc = get_tools()[0].description.lower()
    assert "structured" in desc
    assert "listings" in desc
    assert "broad requests" in desc
    assert "current web" in desc
    assert "specific" in desc and "web_search remains available" in desc
    assert "the source" not in desc


async def test_find_nyc_events_without_toolbox_degrades_to_structured_catalogs(monkeypatch):
    async def fake_ticketmaster(**kwargs):
        return []

    async def fake_dataset(dataset_id, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_dataset)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="what to do in nyc today",
        event_turn="discovery",
    )

    output = await get_tools()[0].handler({}, ctx)

    assert output.startswith(events._NO_RESULTS)
    assert "Current web event leads" not in output


# --- F085: the window generalizes and the named keyword gets a parallel scoped search ---

async def test_window_args_from_the_model_override_the_phrase_window(monkeypatch):
    captured = {}

    async def fake_ticketmaster(**kwargs):
        captured["start_datetime"] = kwargs.get("start_datetime")
        return []

    async def fake_parks(dataset_id, **kwargs):
        captured.setdefault("wheres", []).append(kwargs.get("where"))
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", fake_parks)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]),
                      query="any concerts in early august?")
    await events.get_tools()[0].handler(
        {"keyword": "concerts", "window_start": "2026-08-03", "window_end": "2026-08-05"}, ctx,
    )
    assert captured["start_datetime"] == "2026-08-03T00:00:00Z"
    assert any("startdate >= '2026-08-03'" in w and "2026-08-05" in w for w in captured["wheres"])


def test_find_nyc_events_schema_enforces_documented_dates_and_limit():
    properties = get_tools()[0].parameters["properties"]
    assert properties["window_start"]["format"] == "date"
    assert properties["window_end"]["format"] == "date"
    assert "limit" not in properties
    assert "short noun phrase" in properties["web_query"]["description"].lower()
    assert "every requested constraint" in properties["web_query"]["description"]
    assert properties["setting"]["enum"] == ["indoor", "outdoor"]
