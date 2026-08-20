from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.ticketmaster import TicketmasterSearchResult
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.modules import events
from heynyc.modules.events.tools import (
    Event,
    EventQuery,
    _event_temporal_status,
    _from_parks,
    _temporal_filter,
    _temporal_instruction,
)


def _event(start: str, end: str = "") -> Event:
    return Event(
        name="Test event",
        start_date="2026-08-20",
        start_time=start,
        end_time=end,
        venue="New York",
        borough="Manhattan",
        url="https://example.com/event",
        source="Test",
        tier="editorial",
    )


def test_event_query_uses_dates_and_interval_filters_without_phrase_enums():
    schema = EventQuery.model_json_schema()["properties"]

    assert "relative_window" not in schema
    assert "temporal_scope" not in schema
    assert "weekday" not in schema
    assert {"window_start", "window_end", "window_start_time", "window_end_time"} <= schema.keys()
    assert {"has_started", "has_ended"} <= schema.keys()


def test_event_status_keeps_raw_times_and_computes_relation():
    now = datetime(2026, 8, 20, 14, 24, tzinfo=ZoneInfo("America/New_York"))

    assert _event_temporal_status(_event("5:15 AM", "9:30 AM"), now) == "ended"
    assert _event_temporal_status(_event("2:00 PM", "3:00 PM"), now) == "in_progress"
    assert _event_temporal_status(_event("7:00 PM", "9:00 PM"), now) == "upcoming"
    assert _event_temporal_status(_event("5:15 AM"), now) == "unknown"


def test_event_status_uses_end_date_for_an_overnight_event():
    now = datetime(2026, 8, 21, 1, 0, tzinfo=ZoneInfo("America/New_York"))
    event = _event("11:00 PM", "2:00 AM")
    event.end_date = "2026-08-21"

    assert _event_temporal_status(event, now) == "in_progress"
    assert "ends Friday, 2026-08-21 2:00 AM" in events.tools._event_block(
        event, "S1", now,
    )


def test_parks_adapter_preserves_source_end_values_for_the_shared_evaluation():
    event = _from_parks({
        "title": "Morning concert",
        "startdate": "2026-08-20T00:00:00.000",
        "starttime": "2026-08-20 05:15:00",
        "enddate": "2026-08-20T00:00:00.000",
        "endtime": "2026-08-20 09:30:00",
    })

    assert event is not None
    assert (event.start_date, event.start_time) == ("2026-08-20", "5:15 AM")
    assert (event.end_date, event.end_time) == ("2026-08-20", "9:30 AM")


def test_temporal_filters_compose_without_dropping_unknown_leads():
    now = datetime(2026, 8, 20, 14, 24, tzinfo=ZoneInfo("America/New_York"))
    ended = _event("5:15 AM", "9:30 AM")
    active = _event("2:00 PM", "3:00 PM")
    future = _event("7:00 PM", "9:00 PM")
    unknown = _event("")

    rows = [ended, active, future, unknown]
    assert _temporal_filter(rows, has_started=None, has_ended=False, now=now) == [
        active, future, unknown,
    ]
    assert _temporal_filter(rows, has_started=True, has_ended=False, now=now) == [
        active, unknown,
    ]
    assert _temporal_filter(rows, has_started=None, has_ended=True, now=now) == [ended]


def test_temporal_filters_control_candidate_instructions_too():
    assert "exclude events known to have ended" in _temporal_instruction(None, False)
    assert "in progress" in _temporal_instruction(True, False)
    assert "include only ended events" in _temporal_instruction(None, True)


async def test_generic_event_calendar_gets_one_focused_direct_page_followup(monkeypatch):
    calls = []
    direct_url = (
        "https://www.rockefellercenter.com/events/"
        "raye-live-on-today-at-rockefeller-center"
    )

    async def no_ticketmaster(**_kwargs):
        return TicketmasterSearchResult(status="success", events=[])

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(args, ctx):
        calls.append(("search", args))
        if len(calls) == 1:
            url = "https://www.rockefellercenter.com/events"
            title = "RAYE Live on TODAY at Rockefeller Center"
            snippet = "August 20. Check-in 5:15 a.m.; concert concludes at 9:30 a.m."
            cite = ctx.citations.register(
                url,
                title=title,
                snippet=snippet,
                provenance={"evidence_grade": "discovery"},
            )
            unrelated = ctx.citations.register(
                "https://www.ticketmaster.com/event/unrelated",
                title="Unrelated evening event",
                snippet="August 20 at 8:00 PM",
                provenance={"evidence_grade": "discovery"},
            )
        else:
            url, title, snippet = direct_url, "RAYE Live on TODAY", "Direct event page"
            unrelated = ""
            cite = ctx.citations.register(
                url,
                title=title,
                snippet=snippet,
                provenance={"evidence_grade": "discovery"},
            )
        return f"[{cite}] {title} ({url})\n{snippet}\n{unrelated}"

    async def web_fetch(args, _ctx):
        calls.append(("fetch", args))
        return "Check-in 5:15 a.m.; concert concludes at 9:30 a.m."

    monkeypatch.setattr(events.tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Could you give me a few music events happening in NYC today?",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "search", {"type": "object"}, web_search),
            "web_fetch": Tool("web_fetch", "fetch", {"type": "object"}, web_fetch),
        },
    )

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music", "window_start": "2026-08-20", "has_ended": True}, ctx,
    )

    assert [kind for kind, _args in calls] == ["search", "search", "fetch"]
    assert calls[0][1]["query"] == "NYC Music events August 20, 2026"
    assert calls[1][1]["prefer"] == ["rockefellercenter.com"]
    assert calls[2][1]["url"] == direct_url
    assert direct_url in output


async def test_initial_direct_event_page_is_fetched_without_a_redundant_search(monkeypatch):
    calls = []
    direct_url = "https://example.com/events/direct-concert"

    async def no_ticketmaster(**_kwargs):
        return TicketmasterSearchResult(status="success", events=[])

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(args, ctx):
        calls.append(("search", args))
        direct = ctx.citations.register(
            direct_url, title="Direct concert", snippet="August 20, 7:00 PM",
            provenance={"evidence_grade": "discovery"},
        )
        generic = ctx.citations.register(
            "https://www.eventbrite.com/events", title="Event calendar", snippet="Today",
            provenance={"evidence_grade": "discovery"},
        )
        return f"[{direct}] Direct concert ({direct_url})\n[{generic}] Calendar"

    async def web_fetch(args, _ctx):
        calls.append(("fetch", args))
        return "August 20, 7:00 PM"

    monkeypatch.setattr(events.tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="music today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "search", {"type": "object"}, web_search),
            "web_fetch": Tool("web_fetch", "fetch", {"type": "object"}, web_fetch),
        },
    )

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music", "window_start": "2026-08-20"}, ctx,
    )

    assert [kind for kind, _args in calls] == ["search", "fetch"]
    assert calls[1][1]["url"] == direct_url
    assert direct_url in output


async def test_top_editorial_event_page_is_fetched_without_a_trust_whitelist(monkeypatch):
    calls = []
    url = "https://donyc.com/events/2026/08/20"

    async def no_ticketmaster(**_kwargs):
        return TicketmasterSearchResult(status="success", events=[])

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        cite = ctx.citations.register(
            url, title="Music in NYC", snippet="August 20 events",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        return f"[{cite}] Music in NYC ({url})"

    async def web_fetch(args, _ctx):
        calls.append(args["url"])
        return "Full editorial event page"

    monkeypatch.setattr(events.tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="music today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "search", {"type": "object"}, web_search),
            "web_fetch": Tool("web_fetch", "fetch", {"type": "object"}, web_fetch),
        },
    )

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music", "window_start": "2026-08-20"}, ctx,
    )

    assert calls == [url]
    assert "Full editorial event page" in output
