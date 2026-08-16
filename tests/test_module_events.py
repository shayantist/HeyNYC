from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.prompts import BASE_SYSTEM_PROMPT
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.cases import load_cases
from heynyc.modules.events import tools as event_tools
from heynyc.modules.events.tools import Event, _shortlist


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


def test_event_shortlist_prefers_primary_city_sources_to_ticket_marketplaces() -> None:
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
        "NYC Parks",
        "NYC Permitted Events",
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


def test_event_sources_recognize_the_official_msg_venue_domain() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    module = next(item for item in registry.modules if item.name == "events")

    assert "msg.com" in module.allowlist
    assert "msg.com" in module.source_tiers["authoritative"]
