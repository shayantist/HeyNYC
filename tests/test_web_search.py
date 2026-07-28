from __future__ import annotations

import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools import web_search as web_search_mod
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.web_search import _domain_allowed, web_search_tools

ALLOW = ["nyc.gov", "nyctourism.com", "nynjfwc26.com"]


def test_domain_allowed_matches_subdomains_only():
    assert _domain_allowed("https://www.nyc.gov/events", ALLOW)
    assert _domain_allowed("https://nyctourism.com/x", ALLOW)
    assert not _domain_allowed("https://evil.com/nyc.gov", ALLOW)
    assert not _domain_allowed("https://notnyc.gov/x", ALLOW)  # suffix-spoof guard


async def test_web_search_filters_to_allowlist_and_cites():
    async def fake_search(query, domains, recency=None):
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
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
    }
    assert len(ctx.citations) == 1


async def test_web_search_preserves_shown_evidence_and_explains_when_to_fetch():
    snippet = "x" * 250 + " decisive detail"

    async def fake_search(query, domains, recency=None):
        return [
            {
                "title": "Official",
                "url": "https://www.nyc.gov/snap",
                "snippet": snippet,
            },
        ]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "current SNAP rule"}, ctx)

    assert "decisive detail" in ctx.citations.mapping()["S1"]["snippet"]
    assert "call official_sources" in out
    assert "beyond these snippets" in out


async def test_web_search_marks_archived_results_as_not_current():
    async def fake_search(query, domains, recency=None):
        return [
            {
                "title": "Temporary Protected Status Designated Country",
                "url": (
                    "https://www.uscis.gov/archive/"
                    "temporary-protected-status-designated-country-haiti"
                ),
                "snippet": "DHS announced an older effective date.",
            },
        ]

    tool = web_search_tools(["uscis.gov"], search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await tool.handler({"query": "current Haiti TPS status"}, ctx)

    assert "SOURCE STATUS: ARCHIVED" in out
    assert "SOURCE STATUS: ARCHIVED" in ctx.citations.mapping()["S1"]["snippet"]


async def test_web_search_no_results_abstains():
    async def empty(query, domains, recency=None):
        return []

    tool = web_search_tools(ALLOW, search_fn=empty)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "x"}, ctx)
    assert "couldn't find" in out.lower() or "no results" in out.lower()


async def test_tavily_transport_failure_degrades_to_no_results(monkeypatch):
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._tavily("query", ["nyc.gov"]) == []


async def test_tavily_http_status_failure_is_not_reported_as_no_results(monkeypatch):
    class _Response:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "provider rejected request",
                request=httpx.Request("POST", "https://api.tavily.com/search"),
                response=httpx.Response(429),
            )

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    with pytest.raises(httpx.HTTPStatusError):
        await web_search_mod._tavily("query", ["nyc.gov"])


def _ctx():
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


async def _run(tool, query, **args):
    return await tool.handler({"query": query, **args}, _ctx())


async def test_web_search_ranks_by_tier_and_disclaims_community():
    async def fake_search(query, allowed, recency=None):
        return [
            {"title": "Eventbrite listing", "url": "https://eventbrite.com/e/x", "snippet": "s"},
            {"title": "Official Parks page", "url": "https://nycgovparks.org/events/y", "snippet": "s"},
        ]

    tiers = {"nycgovparks.org": ("authoritative", "events"),
             "eventbrite.com": ("community", "events")}
    allow = ["nycgovparks.org", "eventbrite.com"]
    tool = web_search_tools(allow, source_tiers=tiers, search_fn=fake_search)[0]  # default web_search

    ctx = _ctx()
    out = await tool.handler({"query": "free events"}, ctx)
    # authoritative ranked above community:
    assert out.index("nycgovparks.org") < out.index("eventbrite.com")
    # community carries the disclaimer:
    assert "⚠️" in out
    assert "confirm before you go" in out.lower()
    citations = ctx.citations.mapping()
    assert citations["S1"]["provenance"] == {"evidence_grade": "discovery"}
    assert citations["S2"]["provenance"] == {"evidence_grade": "discovery"}


async def test_web_search_still_hard_gates_ugc_off_allowlist():
    async def fake_search(query, allowed, recency=None):
        return [{"title": "Luma party", "url": "https://lu.ma/x", "snippet": "s"}]

    tool = web_search_tools(["nycgovparks.org"], source_tiers={}, search_fn=fake_search)[0]
    out = await _run(tool, "house party")
    assert "lu.ma" not in out  # off-allowlist domain dropped (defense in depth)


async def test_web_search_prefer_boosts_domain():
    async def fake_search(query, allowed, recency=None):
        return [
            {"title": "Time Out", "url": "https://timeout.com/a", "snippet": "s"},
            {"title": "WorldCup NYC", "url": "https://worldcup.nyc/b", "snippet": "s"},
        ]

    tiers = {"timeout.com": ("editorial", "events"), "worldcup.nyc": ("editorial", "world_cup")}
    tool = web_search_tools(["timeout.com", "worldcup.nyc"], source_tiers=tiers, search_fn=fake_search)[0]
    out = await _run(tool, "watch party", prefer=["worldcup.nyc"])
    assert out.index("worldcup.nyc") < out.index("timeout.com")  # preferred domain first


# --- Currency layer: the recency check (recent_developments) + its subordinate news tier ---

def _by_name(allow, tiers=None, news=None, search_fn=None):
    return {t.name: t for t in web_search_tools(allow, source_tiers=tiers, news_tier=news, search_fn=search_fn)}


async def test_default_web_search_stays_allowlist_only_no_news():
    """Regression guard: the default web_search never unions in the news tier."""
    seen: dict[str, list[str]] = {}

    async def spy(query, domains, recency=None):
        seen[query] = list(domains)
        return [{"title": "Gov", "url": "https://www.nyc.gov/x", "snippet": "s"}]

    tools = _by_name(["nyc.gov"], news=["gothamist.com", "nytimes.com"], search_fn=spy)
    await _run(tools["web_search"], "q")
    assert seen["q"] == ["nyc.gov"]                 # exactly the allowlist
    assert "gothamist.com" not in seen["q"]         # news tier NOT visible to default search


async def test_recent_developments_unions_news_tier():
    """Recency mode searches the trusted allowlist PLUS the curated news tier."""
    seen: dict[str, set[str]] = {}

    async def spy(query, domains, recency=None):
        seen[query] = set(domains)
        return []

    tools = _by_name(["nyc.gov"], news=["gothamist.com", "NYTimes.com"], search_fn=spy)
    out = await _run(tools["recent_developments"], "q")
    assert {"nyc.gov", "gothamist.com", "nytimes.com"} <= seen["q"]  # allowlist ∪ news (lowercased)
    assert "no recent developments" in out.lower()                   # its own abstain message


async def test_recent_developments_ranks_news_below_gov():
    """A news-tier URL ranks below a gov URL, and carries the developing-news label."""
    async def fake(query, domains, recency=None):
        return [
            {"title": "Gothamist story", "url": "https://gothamist.com/a", "snippet": "s"},
            {"title": "Official CCHR", "url": "https://www.nyc.gov/cchr", "snippet": "s"},
        ]

    tools = _by_name(["nyc.gov"], news=["gothamist.com"], search_fn=fake)
    out = await _run(tools["recent_developments"], "voucher ruling")
    assert out.index("nyc.gov") < out.index("gothamist.com")  # gov (authoritative) above news
    assert "📰 news" in out                                    # developing-news label present


async def test_web_search_drops_news_domain_defense_in_depth():
    """Even if the backend slips a news URL into a default web_search, it's dropped."""
    async def fake(query, domains, recency=None):
        return [{"title": "Gothamist", "url": "https://gothamist.com/a", "snippet": "s"}]

    tools = _by_name(["nyc.gov"], news=["gothamist.com"], search_fn=fake)
    out = await _run(tools["web_search"], "q")
    assert "gothamist.com" not in out  # off the (news-free) allowlist → filtered


# --- Agent-settable recency window: what time_range reaches the Tavily backend ---

def _capture_tavily(monkeypatch):
    """Patch the shared `_tavily` and record the kwargs (incl. time_range) each call passes."""
    calls: list[dict] = []

    async def fake_tavily(query, allowed_domains, **extra):
        calls.append(extra)
        return []

    monkeypatch.setattr(web_search_mod, "_tavily", fake_tavily)
    return calls


def _real_backend_tools():
    """Wire the production backends (no injected search_fn) so tavily_search[_recent] run for real."""
    return {t.name: t for t in web_search_tools(["nyc.gov"])}


async def test_recent_developments_recency_week_passes_time_range_week(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["recent_developments"].handler({"query": "q", "recency": "week"}, _ctx())
    assert calls[0]["time_range"] == "week"  # agent-chosen window reaches Tavily


async def test_recent_developments_defaults_to_year(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["recent_developments"].handler({"query": "q"}, _ctx())
    assert calls[0]["time_range"] == "year"  # unset → slow-moving default


async def test_web_search_passes_no_time_range(monkeypatch):
    """Regression guard: the untimed default web_search must never send a time_range."""
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["web_search"].handler({"query": "q"}, _ctx())
    assert "time_range" not in calls[0]


async def test_recent_developments_out_of_enum_recency_falls_back_to_year(monkeypatch):
    """Defense in depth: an unexpected window value falls back to a year, not straight through."""
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["recent_developments"].handler({"query": "q", "recency": "decade"}, _ctx())
    assert calls[0]["time_range"] == "year"


def test_recent_developments_description_warns_on_contested_legal_matter():
    """Red-team MC03/MC04/FP02/ES03 fix: the recency-tool description tells the agent not to restate a
    ruling's court/holding/scope from a news snippet, and to lead with the standing protection instead."""
    tools = {t.name: t for t in web_search_tools(["nyc.gov"], news_tier=["gothamist.com"])}
    desc = tools["recent_developments"].description.lower()
    assert "contested legal matter" in desc
    assert "struck down" in desc and "annulled" in desc
    assert "currently stands" in desc
    assert "never name the court" in desc


def test_rewrite_query_strips_scaffolding_and_errand_verbs():
    """Audited live (F055): searching the resident's prep-phrased sentence returned a
    gardening workshop because 'prepare' matched. The shared rewrite stage normalizes every
    caller's query, the layer where Gemini and ChatGPT put query understanding."""
    from heynyc.core.tools.web_search import _rewrite_query

    assert _rewrite_query("what to prepare for tomorrow wc game July 18, 2026") == (
        "tomorrow wc game July 18, 2026"
    )
    # Civic action verbs are content, never stripped.
    assert _rewrite_query("how do I appeal a SNAP denial") == "appeal a SNAP denial"
    # Nested scaffolding strips iteratively, leading articles too.
    assert _rewrite_query("can you tell me where is the nearest food pantry") == (
        "nearest food pantry"
    )
    # Already search-shaped queries pass through untouched.
    assert _rewrite_query("world cup match schedule July 18 2026") == (
        "world cup match schedule July 18 2026"
    )
    assert _rewrite_query(
        "NYC Section 8 source of income discrimination court ruling 2026"
    ) == "NYC Section 8 source of income discrimination court ruling 2026"
    # Over-stripping falls back to the original instead of searching almost nothing.
    assert _rewrite_query("what to do") == "what to do"


async def test_handler_sends_rewritten_query_and_reports_it():
    seen = []

    async def fake_search(query, domains, recency=None):
        seen.append(query)
        return [{"title": "Official", "url": "https://www.nyc.gov/wc", "snippet": "row"}]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler(
        {"query": "what to prepare for tomorrow wc game July 18, 2026"}, ctx,
    )

    assert seen == ["tomorrow wc game July 18, 2026"]
    # The model sees the effective query, mirroring vendors exposing their search queries.
    assert 'Searched as: "tomorrow wc game July 18, 2026"' in out


async def test_handler_does_not_annotate_unchanged_queries():
    seen = []

    async def fake_search(query, domains, recency=None):
        seen.append(query)
        return [{"title": "Official", "url": "https://www.nyc.gov/wc", "snippet": "row"}]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "world cup schedule"}, ctx)

    assert seen == ["world cup schedule"]
    assert "Searched as" not in out


def test_web_search_defers_nyc_event_listings_to_the_catalog():
    """7/8 tool-choice fix (convo_event_followup_keeps_thread): an events follow-up must route to
    whats_on_events, not web_search. web_search stays the orientation / identity-resolution /
    long-tail tool and hands NYC event listings to the catalog by name; the orientation-first rule
    is preserved."""
    tools = {t.name: t for t in web_search_tools(["nyc.gov"])}
    desc = tools["web_search"].description.lower()
    assert "orientation" in desc and "first" in desc      # orientation-first rule intact
    assert "long-tail" in desc                            # long-tail facts stay here
    assert "whats_on_events" in desc                      # defers NYC event listings to the catalog
    assert "a specific event this weekend" not in desc    # no longer advertises event listings
