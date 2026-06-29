from __future__ import annotations

import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.web_search import _domain_allowed, web_search_tools

ALLOW = ["nyc.gov", "nyctourism.com", "nynjfwc26.com"]


def test_domain_allowed_matches_subdomains_only():
    assert _domain_allowed("https://www.nyc.gov/events", ALLOW)
    assert _domain_allowed("https://nyctourism.com/x", ALLOW)
    assert not _domain_allowed("https://evil.com/nyc.gov", ALLOW)
    assert not _domain_allowed("https://notnyc.gov/x", ALLOW)  # suffix-spoof guard


async def test_web_search_filters_to_allowlist_and_cites():
    async def fake_search(query, domains):
        return [
            {"title": "Official", "url": "https://www.nyc.gov/worldcup", "snippet": "official info"},
            {"title": "Spam", "url": "https://spam.com/x", "snippet": "junk"},  # off-allowlist → dropped
        ]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "world cup nyc"}, ctx)
    assert "Official" in out
    assert "Spam" not in out
    assert ctx.citations.mapping()["S1"]["kind"] == "WEB"
    assert len(ctx.citations) == 1


async def test_web_search_no_results_abstains():
    async def empty(query, domains):
        return []

    tool = web_search_tools(ALLOW, search_fn=empty)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "x"}, ctx)
    assert "couldn't find" in out.lower() or "no results" in out.lower()


def _ctx():
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


async def _run(tool, query, **args):
    return await tool.handler({"query": query, **args}, _ctx())


async def test_web_search_ranks_by_tier_and_disclaims_community():
    async def fake_search(query, allowed):
        return [
            {"title": "Eventbrite listing", "url": "https://eventbrite.com/e/x", "snippet": "s"},
            {"title": "Official Parks page", "url": "https://nycgovparks.org/events/y", "snippet": "s"},
        ]

    tiers = {"nycgovparks.org": ("authoritative", "events"),
             "eventbrite.com": ("community", "events")}
    allow = ["nycgovparks.org", "eventbrite.com"]
    [tool] = web_search_tools(allow, source_tiers=tiers, search_fn=fake_search)

    out = await _run(tool, "free events")
    # authoritative ranked above community:
    assert out.index("nycgovparks.org") < out.index("eventbrite.com")
    # community carries the disclaimer:
    assert "⚠️" in out
    assert "confirm before you go" in out.lower()


async def test_web_search_still_hard_gates_ugc_off_allowlist():
    async def fake_search(query, allowed):
        return [{"title": "Luma party", "url": "https://lu.ma/x", "snippet": "s"}]

    [tool] = web_search_tools(["nycgovparks.org"], source_tiers={}, search_fn=fake_search)
    out = await _run(tool, "house party")
    assert "lu.ma" not in out  # off-allowlist domain dropped (defense in depth)


async def test_web_search_prefer_boosts_domain():
    async def fake_search(query, allowed):
        return [
            {"title": "Time Out", "url": "https://timeout.com/a", "snippet": "s"},
            {"title": "WorldCup NYC", "url": "https://worldcup.nyc/b", "snippet": "s"},
        ]

    tiers = {"timeout.com": ("editorial", "events"), "worldcup.nyc": ("editorial", "world_cup")}
    [tool] = web_search_tools(["timeout.com", "worldcup.nyc"], source_tiers=tiers, search_fn=fake_search)
    out = await _run(tool, "watch party", prefer=["worldcup.nyc"])
    assert out.index("worldcup.nyc") < out.index("timeout.com")  # preferred domain first
