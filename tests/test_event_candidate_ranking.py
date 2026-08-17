from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.web_search import web_search_tools


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
