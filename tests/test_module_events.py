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


def test_events_does_not_silently_correct_source_fields() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    events = next(module for module in registry.modules if module.name == "events")
    prompt = " ".join(events.prompt.lower().split())

    assert "titles and addresses exactly as retrieved" in prompt
    assert "omit a questionable fragment" in prompt


def test_named_event_fact_questions_use_web_search_without_the_shortlist() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    events = next(module for module in registry.modules if module.name == "events")
    prompt = " ".join(events.prompt.lower().split())

    assert "not a request for event choices" in prompt
    assert "call web_search without find_nyc_events" in prompt


def test_web_event_status_uses_the_shared_temporal_evaluator() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    events = next(module for module in registry.modules if module.name == "events")
    prompt = " ".join(events.prompt.lower().split())

    assert "any web-sourced event whose current status you state" in prompt
    assert "call `evaluate_event_time`" in prompt


def test_named_event_time_prefers_the_fetched_official_page() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    events = next(module for module in registry.modules if module.name == "events")
    prompt = " ".join(events.prompt.lower().split())

    assert "fetch the official event page" in prompt
    assert "excerpt lacks the requested time" in prompt
    assert "refetch that page once with the discovered date" in prompt


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


def test_music_classification_does_not_match_an_art_event_that_mentions_music() -> None:
    event = Event(
        "Photography exhibit", "2026-08-21", "9:00 AM", "Central Park", "Manhattan",
        "https://example.com/art", "NYC Parks", "authoritative",
        evidence_excerpt="A photographer documented live music performances.",
        category="Art",
    )

    assert not event_tools._matches_classification(event, "Music")


def test_music_classification_uses_an_excerpt_when_no_category_was_extracted() -> None:
    event = Event(
        "Independent concert", "2026-08-21", "7:00 PM", "Venue", "Brooklyn",
        "https://example.com/concert", "Web discovery", "editorial",
        evidence_excerpt="A music event in New York on August 21.",
    )

    assert event_tools._matches_classification(event, "Music")


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
        {"visit_date": "2099-08-15", "window_end": "2099-08-16"}, ctx,
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
        citation_id = ctx.citations.register(
            "https://organizer.example/events/music-night",
            title="Organizer event page",
            snippet="Music event on August 16, 2099.",
        )
        return f"[{citation_id}] Organizer event page"

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
            "visit_date": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert len(web_calls) == 1
    assert web_calls[0]["query"] == "NYC music events August 16, 2099"
    assert web_calls[0]["count"] == 10
    assert {
        "donyc.com", "eventbrite.com", "luma.com", "ma.to", "nyc.gov", "partiful.com",
    } <= set(web_calls[0]["prefer"])
    assert "123 Main" not in str(web_calls)
    assert "one shared bounded shortlist" in result
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
        cite = ctx.citations.register(
            args["url"],
            title="Official named music concert on the requested date",
            snippet="Official named music concert on the requested date",
            kind="WEB",
            provenance={"evidence_grade": "fetched", "source_tier": "authoritative"},
        )
        return f"SOURCE {cite}: Official named music concert on the requested date"

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
            "visit_date": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert fetched == [{
        "url": "https://venue.example/calendar",
        "query": "NYC Music events August 16, 2099",
    }]
    assert "Official named music concert on the requested date" in result


async def test_topical_event_lookup_fetches_two_ranked_pages_for_structured_events(
    monkeypatch,
) -> None:
    fetched: list[str] = []

    async def no_ticketmaster(**_kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        first = ctx.citations.register(
            "https://timeout.example/music",
            title="Editorial music guide",
            snippet="General NYC event coverage",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        second = ctx.citations.register(
            "https://secretnyc.example/calendar",
            title="NYC events",
            snippet="General city calendar",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        third = ctx.citations.register(
            "https://donyc.example/music",
            title="Music on August 16, 2099",
            snippet="Independent concert at 7 PM on August 16, 2099",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        return f"[{first}] guide\n[{second}] calendar\n[{third}] dated music"

    async def web_fetch(args, ctx):
        fetched.append(args["url"])
        provenance = {"evidence_grade": "fetched", "source_tier": "editorial"}
        if "donyc" in args["url"]:
            provenance["events"] = [{
                "name": "Independent concert",
                "url": "https://donyc.example/events/independent-concert",
                "venue": "Public Records",
                "borough": "Brooklyn",
                "category": "music",
                "start_date": "2099-08-16",
                "start_time": "19:00",
                "end_date": "",
                "end_time": "",
            }]
        cite = ctx.citations.register(
            args["url"], title="Fetched music page", snippet="Fetched music page",
            provenance=provenance,
        )
        return f"SOURCE {cite}: Fetched music page"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="music today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "Search", {}, web_search),
            "web_fetch": Tool("web_fetch", "Fetch", {}, web_fetch),
        },
    )

    result = await event_tools.get_tools()[0].handler(
        {"classification": "Music", "visit_date": "2099-08-16"}, ctx,
    )

    assert fetched == [
        "https://donyc.example/music",
        "https://timeout.example/music",
    ]
    assert "Independent concert" in result
    event_line = next(line for line in result.splitlines() if "Independent concert" in line)
    assert "unconfirmed lead" not in event_line
    event_citation = next(
        citation
        for citation in ctx.citations.mapping().values()
        if citation["url"] == "https://donyc.example/events/independent-concert"
    )
    assert event_citation["kind"] == "DATA"
    assert event_citation["provenance"]["snapshot"]["venue"] == "Public Records"
    assert event_citation["provenance"]["derivation"]["source_citation_id"]


async def test_event_lookup_fetches_declared_daily_calendar_when_search_omits_it(
    monkeypatch,
) -> None:
    fetched: list[str] = []

    async def no_ticketmaster(**_kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        citations = []
        for index in range(10):
            cite = ctx.citations.register(
                f"https://guide{index}.example/music",
                title=f"Music listing {index}",
                snippet=f"Music listing {index}",
                provenance={
                    "evidence_grade": "search_excerpt",
                    "source_tier": "editorial",
                    "event": {
                        "name": f"Search concert {index}",
                        "url": f"https://guide{index}.example/music",
                        "venue": "Brooklyn venue",
                        "borough": "Brooklyn",
                        "category": "music",
                        "start_date": "2099-08-16",
                        "start_time": "20:00",
                    },
                },
            )
            citations.append(f"[{cite}] Music listing {index}")
        return "\n".join(citations)

    async def web_fetch(args, ctx):
        fetched.append(args["url"])
        events = []
        if args["url"] == "https://donyc.com/events/2099/08/16":
            await asyncio.sleep(0.01)
            events = [{
                "name": "Independent concert",
                "url": "https://donyc.com/events/2099/8/16/independent-concert",
                "venue": "Public Records",
                "borough": "Brooklyn",
                "category": "music",
                "start_date": "2099-08-16",
                "start_time": "19:00",
                "end_date": "",
                "end_time": "",
            }]
        cite = ctx.citations.register(
            args["url"], title="Fetched page", snippet="Fetched page",
            provenance={
                "evidence_grade": "fetched", "source_tier": "editorial", "events": events,
                "acquisition": {"fetched_at": "2099-08-16T12:00:00Z"},
            },
        )
        return f"SOURCE {cite}: Fetched page"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    monkeypatch.setattr(event_tools, "_SOURCE_TIMEOUT_S", 0.001)
    monkeypatch.setattr(event_tools, "_PAGE_FETCH_TIMEOUT_S", 0.5)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST),
        query="music today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "Search", {}, web_search),
            "web_fetch": Tool("web_fetch", "Fetch", {}, web_fetch),
        },
    )
    first_result = await event_tools.get_tools()[0].handler(
        {"visit_date": "2099-08-16"}, ctx,
    )
    result = await event_tools.get_tools()[0].handler(
        {"classification": "Music", "visit_date": "2099-08-16"}, ctx,
    )

    assert "https://donyc.com/events/2099/08/16" in fetched
    assert "Independent concert" in first_result
    assert "Independent concert" in result


async def test_calendar_event_with_unknown_locality_is_verified_from_its_direct_page(
    monkeypatch,
) -> None:
    fetched: list[str] = []
    calendar_url = "https://guide.example/music"
    other_url = "https://venue.example/music"
    event_url = "https://guide.example/events/queens-concert"

    async def no_ticketmaster(**_kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        cite = ctx.citations.register(
            calendar_url, title="Music today", snippet="Current music listings",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        other = ctx.citations.register(
            other_url, title="Another music page", snippet="Current music listings",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        child = ctx.citations.register(
            event_url, title="Queens concert", snippet="Music today at 7 PM",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        return f"[{cite}] Music today\n[{other}] Another page\n[{child}] Queens concert"

    async def web_fetch(args, ctx):
        fetched.append(args["url"])
        direct = args["url"] == event_url
        calendar = args["url"] == calendar_url
        event = {
            "name": "Queens concert",
            "url": event_url,
            "venue": "Forest Hills Stadium",
            "borough": "Forest Hills" if direct else "",
            "category": "music",
            "start_date": "2099-08-16",
            "start_time": "19:00",
            "end_date": "",
            "end_time": "",
        }
        cite = ctx.citations.register(
            args["url"], title="Queens concert", snippet="Queens concert",
            provenance={
                "evidence_grade": "fetched", "source_tier": "editorial",
                **({"events": [event]} if direct or calendar else {}),
            },
        )
        return f"SOURCE {cite}: Queens concert"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="music today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "Search", {}, web_search),
            "web_fetch": Tool("web_fetch", "Fetch", {}, web_fetch),
        },
    )

    result = await event_tools.get_tools()[0].handler(
        {"classification": "Music", "visit_date": "2099-08-16"}, ctx,
    )

    assert fetched == [calendar_url, other_url, event_url]
    assert "Queens concert" in result
    assert "Forest Hills" in result


async def test_general_event_discovery_fetches_an_exact_date_calendar(monkeypatch) -> None:
    fetched: list[str] = []

    async def no_ticketmaster(**_kwargs):
        return event_tools.TicketmasterSearchResult(status="complete")

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        cite = ctx.citations.register(
            "https://events.example/2099/08/16",
            title="NYC events on August 16, 2099",
            snippet="Events happening on August 16, 2099",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
        )
        return f"[{cite}] dated calendar"

    async def web_fetch(args, _ctx):
        fetched.append(args["url"])
        return "Fetched calendar"

    monkeypatch.setattr(event_tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(event_tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="what to do today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "Search", {}, web_search),
            "web_fetch": Tool("web_fetch", "Fetch", {}, web_fetch),
        },
    )

    await event_tools.get_tools()[0].handler({"visit_date": "2099-08-16"}, ctx)

    assert fetched == ["https://events.example/2099/08/16"]


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
            "visit_date": "2099-08-16",
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


def test_event_sources_include_current_approachable_discovery_domains() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    module = next(item for item in registry.modules if item.name == "events")

    discovery_domains = {
        domain
        for tier in ("editorial", "community")
        for domain in module.source_tiers[tier]
    }

    assert {"donyc.com", "eventbrite.com", "luma.com", "ma.to", "partiful.com"} <= (
        discovery_domains
    )


def test_event_tool_schema_comes_from_the_typed_query_model() -> None:
    schema = event_tools.get_tools()[0].parameters

    assert schema["title"] == "EventQuery"
    assert schema["additionalProperties"] is False


def test_event_query_rejects_a_reversed_date_window() -> None:
    with pytest.raises(ValueError, match="window_end must not be before visit_date"):
        EventQuery.model_validate({
            "visit_date": "2099-08-17",
            "window_end": "2099-08-16",
        })


def test_event_query_owns_date_time_and_cost_constraints() -> None:
    query = EventQuery.model_validate({
        "visit_date": "2026-08-18",
        "visit_time": "17:00:00",
        "cost": "free",
    })

    assert query.visit_date.isoformat() == "2026-08-18"
    assert query.visit_time.isoformat() == "17:00:00"
    assert query.cost == "free"


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
        {"visit_date": "2026-08-17"}, ctx,
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
            "visit_date": "2099-08-17",
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

    await event_tools.get_tools()[0].handler({"visit_date": "2099-08-20"}, ctx)

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
        cite = ctx.citations.register(
            "https://community.example/web-concert",
            title="Web music concert",
            snippet="Community-posted music concert on August 16.",
            provenance={"evidence_grade": "search_excerpt", "source_tier": "community"},
        )
        return f"[{cite}] (community-posted, confirm before you go) Web music concert"

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
            "visit_date": "2099-08-16",
            "window_end": "2099-08-16",
        },
        ctx,
    )

    assert "Current web event leads" not in output
    assert "one shared bounded shortlist" in output
    assert "Web music concert" in output
    assert "Catalog concert" in output
