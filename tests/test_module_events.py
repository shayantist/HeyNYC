from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
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


async def test_event_sources_do_not_truncate_before_shortlisting(monkeypatch) -> None:
    limits = []

    async def no_ticketmaster(**kwargs):
        return []

    async def capture_query(*args, **kwargs):
        limits.append(kwargs.get("limit"))
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

    assert limits == [None, None]
