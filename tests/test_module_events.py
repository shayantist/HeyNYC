import asyncio

import pytest

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.prompts import BASE_SYSTEM_PROMPT
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.cases import load_cases
from heynyc.modules.events import tools as event_tools
from heynyc.modules.events.tools import Event, EventQuery, _shortlist


def test_ended_world_cup_case_uses_open_web_orientation() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    case = next(
        case for case in load_cases(registry)
        if case.id == "events_abbreviated_game_preparation"
    )

    assert case.expect_tools == ["web_search"]


def test_events_does_not_present_partial_constraint_matches_as_options() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    events = next(module for module in registry.modules if module.name == "events")
    prompt = " ".join(events.prompt.lower().split())

    assert "only some requested constraints" in prompt
    assert "do not present it as a matching option" in prompt
    assert "official live-listing page" in prompt


def test_event_shortlist_preserves_each_requested_date() -> None:
    events = [
        Event(
            f"Saturday event {index}", "2026-08-15", "", "", "", f"sat-{index}",
            "NYC Parks", "authoritative",
        )
        for index in range(20)
    ] + [
        Event(
            f"Sunday event {index}", "2026-08-16", "", "", "", f"sun-{index}",
            "NYC Parks", "authoritative",
        )
        for index in range(9)
    ]

    shortlisted = _shortlist(events, 20)

    assert {event.start_date for event in shortlisted} == {"2026-08-15", "2026-08-16"}


def test_event_shortlist_uses_event_time_not_source_as_the_tiebreaker() -> None:
    events = [
        Event(
            "Marketplace concert", "2026-08-16", "8:00 PM", "Club", "Brooklyn", "tm",
            "Ticketmaster Discovery", "authoritative",
        ),
        Event(
            "Parks concert", "2026-08-16", "6:00 PM", "Central Park", "Manhattan", "parks",
            "NYC Parks", "authoritative",
        ),
        Event(
            "Street festival", "2026-08-16", "5:00 PM", "Queens Plaza", "Queens", "permit",
            "NYC Permitted Events", "authoritative",
        ),
    ]

    shortlisted = _shortlist(events, 2)

    assert [event.source for event in shortlisted] == [
        "NYC Permitted Events",
        "NYC Parks",
    ]


def test_list_guidance_discloses_default_shortlists_and_offers_a_next_step() -> None:
    guidance = " ".join(BASE_SYSTEM_PROMPT.lower().split())

    assert "make clear it is a shortlist" in guidance
    assert "offer a useful next step" in guidance


async def test_event_sources_do_not_truncate_before_shortlisting(monkeypatch) -> None:
    queries = []

    async def no_ticketmaster(**kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def capture_query(*args, **kwargs):
        queries.append((kwargs.get("limit"), kwargs.get("offset"), kwargs.get("order")))
        return []

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", capture_query)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="free events this weekend",
        event_turn="discovery",
    )

    await event_tools.get_tools()[0].handler(
        {"window_start": "2099-08-15", "window_end": "2099-08-16"}, ctx,
    )

    assert queries == [
        (1000, 0, "startdate, :id"),
        (1000, 0, "start_date_time, :id"),
    ]


async def test_topical_event_lookup_also_searches_for_primary_sources(monkeypatch) -> None:
    web_calls = []

    async def no_ticketmaster(**kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_search(args, ctx):
        web_calls.append(args)
        return "Organizer event page"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST),
        query="Anything music related?",
        user_turns=("I live at 123 Main St. What is on today?", "Anything music related?"),
        event_turn="discovery",
        toolbox={
            "web_search": Tool(
                "web_search",
                "Search the web",
                {"type": "object", "properties": {}},
                web_search,
            )
        },
    )

    result = await event_tools.get_tools()[0].handler(
        {
            "keyword": "music",
            "classification": "Music",
            "window_start": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert web_calls == [{
        "query": "NYC music events August 16, 2099",
        "count": 10,
    }]
    assert "123 Main" not in str(web_calls)
    assert "Candidate event choices" in result
    assert "Choose by exact date and topic match first" in result


async def test_topical_event_lookup_fetches_the_first_authoritative_search_page(
    monkeypatch,
) -> None:
    fetched = []

    async def no_ticketmaster(**kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_search(args, ctx):
        cite = ctx.citations.register(
            "https://venue.example/calendar",
            title="Venue calendar",
            snippet="Upcoming music at the venue",
            kind="WEB",
            provenance={"evidence_grade": "authoritative_excerpt"},
        )
        return f"[{cite}] (authoritative) Venue calendar"

    async def web_fetch(args, ctx):
        await asyncio.sleep(0.01)
        fetched.append(args)
        return "SOURCE: Official named concert on the requested date"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    monkeypatch.setattr(event_tools, "_SOURCE_TIMEOUT_S", 0.001)
    monkeypatch.setattr(event_tools, "_PAGE_FETCH_TIMEOUT_S", 0.05)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Anything music related?",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "Search", {}, web_search),
            "web_fetch": Tool("web_fetch", "Fetch", {}, web_fetch),
        },
    )

    result = await event_tools.get_tools()[0].handler(
        {
            "classification": "Music",
            "window_start": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert fetched == [{
        "url": "https://venue.example/calendar",
        "query": "NYC Music events August 16, 2099",
    }]
    assert "Official named concert on the requested date" in result


async def test_music_category_uses_ticketmaster_classification_and_exact_window(monkeypatch) -> None:
    ticketmaster_calls = []

    async def capture_ticketmaster(**kwargs):
        ticketmaster_calls.append(kwargs)
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*args, **kwargs):
        return []

    monkeypatch.setattr(event_tools, "ticketmaster_events", capture_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Anything music related?",
        event_turn="discovery",
    )

    await event_tools.get_tools()[0].handler(
        {
            "keyword": "music",
            "classification": "Music",
            "window_start": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert ticketmaster_calls[0]["keyword"] is None
    assert ticketmaster_calls[0]["classification"] == "Music"
    assert ticketmaster_calls[0]["start_datetime"] == "2099-08-16T04:00:00Z"
    assert ticketmaster_calls[0]["end_datetime"] == "2099-08-17T04:00:00Z"
    assert ticketmaster_calls[0]["size"] == 200


def test_event_sources_recognize_the_official_msg_venue_domain() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    module = next(item for item in registry.modules if item.name == "events")

    assert "msg.com" in module.allowlist
    assert "msg.com" in module.source_tiers["authoritative"]


def test_event_tool_schema_comes_from_the_typed_query_model() -> None:
    schema = event_tools.get_tools()[0].parameters

    assert schema["title"] == "EventQuery"
    assert schema["additionalProperties"] is False


def test_event_query_rejects_a_reversed_date_window() -> None:
    with pytest.raises(ValueError, match="window_end must not be before window_start"):
        EventQuery.model_validate({
            "window_start": "2099-08-17",
            "window_end": "2099-08-16",
        })


def test_event_query_does_not_mix_relative_and_absolute_windows() -> None:
    with pytest.raises(ValueError, match="relative_window cannot be combined"):
        EventQuery.model_validate({
            "relative_window": "this_weekend",
            "window_start": "2099-08-17",
        })


def test_event_query_owns_relative_time_and_cost_constraints() -> None:
    query = EventQuery.model_validate({
        "relative_window": "tomorrow",
        "cost": "free",
    })

    assert query.relative_window == "tomorrow"
    assert query.cost == "free"
    assert event_tools._relative_window("tomorrow", "2026-08-17") == (
        "2026-08-18", "2026-08-18",
    )
    assert event_tools._relative_window("this_weekend", "2026-08-22") == (
        "2026-08-22", "2026-08-23",
    )
    assert event_tools._relative_window("this_weekend", "2026-08-23") == (
        "2026-08-23", "2026-08-23",
    )


async def test_event_tool_does_not_reinterpret_raw_resident_constraints(monkeypatch) -> None:
    class FixedDateTime(event_tools.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 17, 15, 0, tzinfo=tz)

    async def no_ticketmaster(**kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def city_rows(dataset_id, **kwargs):
        if dataset_id != event_tools.PARKS_DATASET_ID:
            return []
        return [
            {
                "title": "Afternoon drawing",
                "startdate": "2026-08-17",
                "starttime": "16:00:00",
                "link": "https://example.com/drawing",
            },
            {
                "title": "Free evening music",
                "description": "This concert is free.",
                "startdate": "2026-08-17",
                "starttime": "19:00:00",
                "link": "https://example.com/music",
            },
        ]

    monkeypatch.setattr(event_tools, "datetime", FixedDateTime)
    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="Show me free events tonight",
    )

    output = await event_tools.get_tools()[0].handler(
        {"relative_window": "today"}, ctx,
    )

    assert "Afternoon drawing" in output
    assert "Free evening music" in output


async def test_event_tool_applies_typed_free_constraint(monkeypatch) -> None:
    async def no_ticketmaster(**kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def city_rows(dataset_id, **kwargs):
        if dataset_id != event_tools.PARKS_DATASET_ID:
            return []
        return [
            {
                "title": "Unknown-price drawing",
                "startdate": "2099-08-17",
                "starttime": "16:00:00",
                "link": "https://example.com/drawing",
            },
            {
                "title": "Free evening music",
                "description": "This concert is free.",
                "startdate": "2099-08-17",
                "starttime": "19:00:00",
                "link": "https://example.com/music",
            },
        ]

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="events",
    )

    output = await event_tools.get_tools()[0].handler(
        {
            "window_start": "2099-08-17",
            "window_end": "2099-08-17",
            "cost": "free",
        },
        ctx,
    )

    assert "Unknown-price drawing" not in output
    assert "Free evening music" in output


async def test_single_absolute_event_date_is_not_an_open_ended_window(monkeypatch) -> None:
    calls = []

    async def capture_ticketmaster(**kwargs):
        calls.append(kwargs)
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*args, **kwargs):
        return []

    monkeypatch.setattr(event_tools, "ticketmaster_events", capture_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="events on August 20",
    )

    await event_tools.get_tools()[0].handler({"window_start": "2099-08-20"}, ctx)

    assert calls[0]["start_datetime"] == "2099-08-20T04:00:00Z"
    assert calls[0]["end_datetime"] == "2099-08-21T04:00:00Z"


async def test_web_and_catalog_events_share_one_ranked_candidate_pool(monkeypatch) -> None:
    async def one_ticketmaster_event(**kwargs):
        return event_tools.TicketmasterSearchResult(
            status="complete",
            events=[{
                "name": "Catalog concert",
                "url": "https://tickets.example/catalog",
                "dates": {"start": {"localDate": "2099-08-16", "localTime": "20:00:00"}},
                "_embedded": {"venues": [{"name": "Club", "city": {"name": "New York"}}]},
            }],
        )

    async def no_city_rows(*args, **kwargs):
        return []

    async def web_search(args, ctx):
        return "[S1] (community-posted, confirm before you go) Web concert"

    monkeypatch.setattr(event_tools, "ticketmaster_events", one_ticketmaster_event)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="music on August 16",
        event_turn="discovery",
        toolbox={
            "web_search": Tool(
                "web_search",
                "Search the web",
                {"type": "object", "properties": {}},
                web_search,
            )
        },
    )

    output = await event_tools.get_tools()[0].handler(
        {
            "classification": "Music",
            "window_start": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert "Current web event leads" not in output
    assert "Rank every candidate together" in output
    assert output.index("Web concert") < output.index("Catalog concert")
