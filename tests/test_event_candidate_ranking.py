from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.ticketmaster import TicketmasterSearchResult
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.web_search import web_search_tools
from heynyc.modules import events
from heynyc.modules.events.tools import (
    Event,
    _event_block,
    _from_web_citation,
    _shortlist,
)


def test_web_citation_normalizes_into_the_existing_event_record() -> None:
    event = _from_web_citation(
        "S4",
        {
            "url": "https://donyc.com/events/2026/08/21/show",
            "title": "A neighborhood concert",
            "snippet": "Friday night at Public Records in Brooklyn.",
            "provenance": {
                "source_tier": "editorial",
                "search": {"provider": "Tavily Search API", "score": 0.91},
            },
        },
        rank=2,
    )

    assert event == Event(
        name="A neighborhood concert",
        start_date="",
        start_time="",
        venue="",
        borough="",
        url="https://donyc.com/events/2026/08/21/show",
        source="Web discovery",
        tier="editorial",
        publishing_source="donyc.com",
        provider_id="https://donyc.com/events/2026/08/21/show",
        provider_record={
            "url": "https://donyc.com/events/2026/08/21/show",
            "title": "A neighborhood concert",
            "snippet": "Friday night at Public Records in Brooklyn.",
            "provenance": {
                "source_tier": "editorial",
                "search": {"provider": "Tavily Search API", "score": 0.91},
            },
        },
        evidence_excerpt="Friday night at Public Records in Brooklyn.",
        citation_id="S4",
        retrieval_rank=2,
    )


def test_unified_shortlist_keeps_a_web_candidate_beside_catalog_candidates() -> None:
    rows = [
        Event(
            name=f"Ticketmaster {index}", start_date="2026-08-21", start_time="7:00 PM",
            venue=f"Venue {index}", borough="Manhattan", url=f"https://ticketmaster.test/{index}",
            source="Ticketmaster Discovery", tier="authoritative",
        )
        for index in range(5)
    ]
    rows += [
        Event(
            name="DoNYC pick", start_date="", start_time="", venue="", borough="",
            url="https://donyc.test/pick", source="Web discovery", tier="editorial",
            citation_id="S9", retrieval_rank=0,
        )
    ]

    shortlist = _shortlist(rows, 5)

    assert len(shortlist) == 5
    assert any(event.name == "DoNYC pick" for event in shortlist)


def test_incomplete_community_lead_keeps_constraint_and_trust_warnings() -> None:
    event = Event(
        name="Community concert", start_date="", start_time="", venue="", borough="",
        url="https://luma.com/community-concert", source="Web discovery", tier="community",
    )

    block = _event_block(event, "S1")

    assert "unconfirmed lead, not a matching option" in block
    assert "community-posted, confirm before you go" in block


def test_structured_editorial_event_is_not_labeled_an_unconfirmed_lead() -> None:
    event = Event(
        name="Independent concert", start_date="2026-08-21", start_time="17:30",
        venue="Forest Hills Stadium", borough="Queens",
        url="https://donyc.com/events/independent-concert",
        source="Web discovery", tier="editorial", structured_source=True,
    )

    block = _event_block(event, "S1")

    assert "unconfirmed lead" not in block
    assert "editorial source, cite only what the excerpt states" in block


async def test_event_handler_sends_one_bounded_mixed_candidate_list_to_the_model(
    monkeypatch,
) -> None:
    async def ticketmaster(**_kwargs):
        return TicketmasterSearchResult(
            status="complete",
            events=[
                {
                    "id": f"tm-{index}",
                    "name": f"Marketplace concert {index}",
                    "url": f"https://ticketmaster.test/{index}",
                    "dates": {"start": {"localDate": "2026-08-21", "localTime": "19:00:00"}},
                    "_embedded": {"venues": [{"name": f"Venue {index}", "city": {"name": "New York"}}]},
                }
                for index in range(6)
            ],
        )

    async def no_city_rows(*_args, **_kwargs):
        return []

    async def web_search(_args, ctx):
        blocks = []
        for index in range(8):
            citation_id = ctx.citations.register(
                f"https://donyc.test/event-{index}",
                title=f"Independent concert {index}",
                snippet="A music event in New York on August 21.",
                provenance={"evidence_grade": "search_excerpt", "source_tier": "editorial"},
            )
            blocks.append(f"[{citation_id}] Independent concert {index}")
        return "\n".join(blocks)

    monkeypatch.setattr(events.tools, "ticketmaster_events", ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", no_city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        query="music events today",
        event_turn="discovery",
        toolbox={
            "web_search": Tool("web_search", "search", {"type": "object"}, web_search),
        },
    )

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music", "window_start": "2026-08-21"},
        ctx,
    )

    assert "Independent concert 0" in output
    assert "Marketplace concert 0" in output, output
    assert "Independent concert 7" not in output
    assert "Web-discovered candidates:" not in output
    assert output.count("\n- ") <= 5


async def test_classification_filters_every_candidate_source_before_ranking(
    monkeypatch,
) -> None:
    async def no_ticketmaster(**_kwargs):
        return TicketmasterSearchResult(status="complete")

    async def city_rows(dataset_id, **_kwargs):
        if dataset_id != events.tools.PARKS_DATASET_ID:
            return []
        return [
            {
                "title": "Morning pickleball",
                "startdate": "2099-08-21",
                "link": "https://parks.example/pickleball",
            },
            {
                "title": "Evening music concert",
                "startdate": "2099-08-21",
                "link": "https://parks.example/concert",
            },
        ]

    async def web_search(_args, ctx):
        generic = ctx.citations.register(
            "https://guide.example/today",
            title="Things to do today",
            snippet="A citywide activity guide.",
        )
        music = ctx.citations.register(
            "https://guide.example/music",
            title="Music tonight",
            snippet="Concerts in NYC on August 21.",
        )
        return f"[{generic}] Things to do today\n[{music}] Music tonight"

    monkeypatch.setattr(events.tools, "ticketmaster_events", no_ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", city_rows)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), query="music today",
        event_turn="discovery",
        toolbox={"web_search": Tool("web_search", "search", {"type": "object"}, web_search)},
    )

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music", "window_start": "2099-08-21"}, ctx,
    )

    assert "Evening music concert" in output
    assert "Music tonight" in output
    assert "Morning pickleball" not in output
    assert "Things to do today" not in output


async def test_no_date_defaults_to_the_current_new_york_day(monkeypatch) -> None:
    class FixedDateTime(events.tools.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 21, 9, 0, tzinfo=tz)

    async def ticketmaster(**_kwargs):
        return TicketmasterSearchResult(
            status="complete",
            events=[
                {"name": "Today concert", "dates": {"start": {"localDate": "2026-08-21"}}},
                {"name": "Tomorrow concert", "dates": {"start": {"localDate": "2026-08-22"}}},
            ],
        )

    async def no_city_rows(*_args, **_kwargs):
        return []

    monkeypatch.setattr(events.tools, "datetime", FixedDateTime)
    monkeypatch.setattr(events.tools, "ticketmaster_events", ticketmaster)
    monkeypatch.setattr(events.tools, "query_dataset", no_city_rows)

    output = await events.tools.get_tools()[0].handler(
        {"classification": "Music"},
        ToolContext(citations=CitationRegistry(), registry=Registry([]), query="music"),
    )

    assert "Today concert" in output
    assert "Tomorrow concert" not in output


async def test_event_relevance_ranking_does_not_demote_community_sources() -> None:
    async def search(*args, **kwargs):
        return [
            {
                "title": "Exact Luma concert",
                "url": "https://lu.ma/exact-concert",
                "snippet": "Music in NYC on August 17, 2026",
                "score": 0.95,
            },
            {
                "title": "Generic parks calendar",
                "url": "https://nycgovparks.org/events",
                "snippet": "Browse events",
                "score": 0.20,
            },
        ]

    tool = web_search_tools(
        ["nycgovparks.org", "lu.ma"],
        source_tiers={
            "nycgovparks.org": ("authoritative", "events"),
            "lu.ma": ("community", "events"),
        },
        search_fn=search,
    )[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        event_turn="discovery",
    )

    output = await tool.handler(
        {
            "query": "music events NYC August 17 2026",
            "count": 10,
        },
        ctx,
    )

    assert output.index("Exact Luma concert") < output.index("Generic parks calendar")
    assert "community-posted, confirm before you go" in output


async def test_non_event_search_still_ranks_evidence_trust_first() -> None:
    async def search(*args, **kwargs):
        return [
            {
                "title": "Exact community claim",
                "url": "https://example.com/exact",
                "snippet": "Exact but unverified",
                "score": 0.95,
            },
            {
                "title": "Official claim",
                "url": "https://nyc.gov/official",
                "snippet": "Official city information",
                "score": 0.20,
            },
        ]

    tool = web_search_tools(["nyc.gov", "example.com"], search_fn=search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    output = await tool.handler({"query": "benefit eligibility", "count": 10}, ctx)

    assert output.index("Official claim") < output.index("Exact community claim")


async def test_high_stakes_event_search_still_ranks_evidence_trust_first() -> None:
    async def search(*args, **kwargs):
        return [
            {
                "title": "Community event claim",
                "url": "https://example.com/event",
                "snippet": "Unverified event accessibility claim",
                "score": 0.99,
            },
            {
                "title": "Official event claim",
                "url": "https://nyc.gov/event",
                "snippet": "Official city accessibility information",
                "score": 0.01,
            },
        ]

    tool = web_search_tools(["nyc.gov", "example.com"], search_fn=search)[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        event_turn="discovery",
        current_turn_high_stakes=True,
    )

    output = await tool.handler({"query": "accessible event", "count": 10}, ctx)

    assert output.index("Official event claim") < output.index("Community event claim")
