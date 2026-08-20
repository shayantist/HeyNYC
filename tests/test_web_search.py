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


async def test_web_search_keeps_unlisted_results_and_marks_them_unverified():
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [
            {"title": "Official", "url": "https://www.nyc.gov/worldcup", "snippet": "official info"},
            {"title": "Unlisted", "url": "https://example.com/x", "snippet": "lead"},
        ]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "world cup nyc"}, ctx)
    assert "Official" in out
    assert "Unlisted" in out
    assert "search excerpt, cite only what it states" in out.lower()
    assert ctx.citations.mapping()["S1"]["kind"] == "WEB"
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "authoritative_excerpt",
        "source_tier": "authoritative",
    }
    assert ctx.citations.mapping()["S2"]["provenance"] == {
        "evidence_grade": "discovery",
        "source_tier": "unverified",
    }
    assert len(ctx.citations) == 2


async def test_web_search_returns_evidence_without_answer_policy():
    snippet = "x" * 250 + " decisive detail"

    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
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
    assert "call web_fetch" not in out
    assert "details beyond an excerpt" not in out


async def test_web_search_prefers_a_canonical_page_without_merging_variant_evidence():
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [
            {
                "title": "Request Hearing",
                "url": "https://otda.ny.gov/hearings/request/",
                "snippet": "You may request a Fair Hearing online.",
            },
            {
                "title": "Fair Hearing portal",
                "url": "https://otda.ny.gov/hearings/request/?vertical=SSDI",
                "snippet": "A Fair Hearing lets you challenge a local agency decision.",
            },
        ]

    tool = web_search_tools(["otda.ny.gov"], search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await tool.handler({"query": "request a Fair Hearing"}, ctx)

    assert len(ctx.citations) == 1
    citation = ctx.citations.mapping()["S1"]
    assert citation["url"] == "https://otda.ny.gov/hearings/request/"
    assert "request a Fair Hearing online" in citation["snippet"]
    assert "challenge a local agency decision" not in citation["snippet"]
    assert "vertical=SSDI" not in out


async def test_web_search_keeps_distinct_query_specific_pages_separate():
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [
            {
                "title": "Program details",
                "url": "https://example.org/program?id=one",
                "snippet": "Details for program one.",
            },
            {
                "title": "Program details",
                "url": "https://example.org/program?id=two",
                "snippet": "Details for program two.",
            },
        ]

    tool = web_search_tools(["example.org"], search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    await tool.handler({"query": "program details"}, ctx)

    assert {citation["url"] for citation in ctx.citations.mapping().values()} == {
        "https://example.org/program?id=one",
        "https://example.org/program?id=two",
    }


async def test_web_search_marks_an_official_excerpt_as_bounded_answer_evidence():
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [
            {
                "title": "Knicks Name Jalen Brunson Team Captain",
                "url": "https://www.nba.com/knicks/news/team-captain",
                "snippet": "The New York Knicks named Jalen Brunson the 36th captain in franchise history.",
            },
        ]

    tool = web_search_tools(
        ["nba.com"],
        source_tiers={"nba.com": ("authoritative", "events")},
        search_fn=fake_search,
    )[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await tool.handler({"query": "current Knicks captain"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "authoritative_excerpt",
        "source_tier": "authoritative",
    }
    assert "only claims directly supported by an official excerpt" not in out.lower()


@pytest.mark.parametrize("tier", ["editorial", "news"])
async def test_web_search_marks_curated_excerpts_as_bounded_answer_evidence(tier):
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        return [{
            "title": "Knicks captain",
            "url": "https://example-news.com/knicks",
            "snippet": "Jalen Brunson is the captain of the New York Knicks.",
        }]

    tool = web_search_tools(
        ["example-news.com"],
        source_tiers={"example-news.com": (tier, "events")},
        search_fn=fake_search,
    )[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await tool.handler({"query": "current Knicks captain"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "search_excerpt",
        "source_tier": tier,
    }
    assert "cite only what the excerpt states" in out.lower()


async def test_web_search_marks_archived_results_as_not_current():
    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
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
    async def empty(query, domains, published_after=None, published_before=None, count=5):
        return []

    tool = web_search_tools(ALLOW, search_fn=empty)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler({"query": "x"}, ctx)
    assert "couldn't find" in out.lower() or "no results" in out.lower()
    assert "tell the user" not in out.lower()
    assert "rather than guessing" not in out.lower()


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


async def test_tavily_basic_search_does_not_restrict_results_to_known_domains(monkeypatch):
    request_json = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **kwargs):
            request_json.update(kwargs["json"])
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    await web_search_mod._tavily("things to do in NYC today", ["nyc.gov"])

    assert request_json["search_depth"] == "basic"
    assert "include_domains" not in request_json


async def test_tavily_preserves_result_score_and_publication_date(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{
                "title": "Current status",
                "url": "https://example.com/current",
                "content": "The current status is active.",
                "score": 0.91,
                "published_date": "2026-08-14T10:30:00Z",
            }]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._tavily("current status", []) == [{
        "title": "Current status",
        "url": "https://example.com/current",
        "snippet": "The current status is active.",
        "search_provider": "Tavily Search API",
        "score": 0.91,
        "published_date": "2026-08-14T10:30:00Z",
    }]


async def test_web_search_exposes_tavily_metadata_and_uses_score_as_a_tiebreaker():
    async def fake_search(
        query, domains, published_after=None, published_before=None, count=5,
    ):
        return [
            {
                "title": "Lower relevance",
                "url": "https://www.nyc.gov/lower",
                "snippet": "Older support.",
                "search_provider": "Tavily Search API",
                "score": 0.61,
                "published_date": "2026-08-01T10:00:00Z",
            },
            {
                "title": "Higher relevance",
                "url": "https://www.nyc.gov/higher",
                "snippet": "Current support.",
                "search_provider": "Tavily Search API",
                "score": 0.94,
                "published_date": "2026-08-14T10:00:00Z",
            },
        ]

    ctx = _ctx()
    out = await web_search_tools(["nyc.gov"], search_fn=fake_search)[0].handler(
        {"query": "current status"}, ctx,
    )

    assert out.index("Higher relevance") < out.index("Lower relevance")
    assert "Published: 2026-08-14T10:00:00Z" in out
    assert "Provider relevance score: 0.94 (not truth confidence)" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["search"] == {
        "provider": "Tavily Search API",
        "score": 0.94,
        "published_date": "2026-08-14T10:00:00Z",
    }


async def test_web_search_reports_degraded_provider_scope_once():
    async def degraded_search(
        query, domains, published_after=None, published_before=None, count=5, topic=None,
    ):
        return [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": "Current lead",
                "search_provider": "DuckDuckGo",
                "degraded_from_topic": "news",
            }
            for index in range(2)
        ]

    ctx = _ctx()
    out = await web_search_tools([], search_fn=degraded_search)[0].handler(
        {"query": "current reporting", "topic": "news"}, ctx,
    )

    assert out.count("Search provider: DuckDuckGo") == 1
    assert out.count("general fallback without provider publication-date guarantees") == 1
    assert all(
        citation["provenance"]["search"]["degraded_from_topic"] == "news"
        for citation in ctx.citations.mapping().values()
    )


async def test_web_search_exposes_news_publisher_metadata():
    async def news_search(
        query, domains, published_after=None, published_before=None, count=5, topic=None,
    ):
        return [{
            "title": "Current captain interview",
            "url": "https://example.com/current-captain",
            "snippet": "The current captain discussed the coming season.",
            "search_provider": "DuckDuckGo News",
            "published_date": "2026-08-15T11:30:00+00:00",
            "publisher": "Example Sports",
        }]

    ctx = _ctx()
    out = await web_search_tools([], search_fn=news_search)[0].handler(
        {"query": "current captain", "topic": "news"}, ctx,
    )

    assert "Publisher: Example Sports" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["search"] == {
        "provider": "DuckDuckGo News",
        "published_date": "2026-08-15T11:30:00+00:00",
        "publisher": "Example Sports",
    }


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


async def test_search_falls_back_to_duckduckgo_when_tavily_plan_is_exhausted(monkeypatch):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def fallback(query, allowed_domains, count=5, topic=None):
        assert query == "current NYC service"
        assert allowed_domains == ["nyc.gov"]
        assert count == 7
        assert topic is None
        return [{"title": "Official", "url": "https://nyc.gov/service", "snippet": "Current"}]

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", fallback)

    assert await web_search_mod.tavily_search(
        "current NYC service", ["nyc.gov"], count=7
    ) == [{
        "title": "Official",
        "url": "https://nyc.gov/service",
        "snippet": "Current",
        "search_provider": "DuckDuckGo",
    }]


async def test_duckduckgo_news_fallback_preserves_publication_metadata(monkeypatch):
    import ddgs

    class FakeDDGS:
        def news(self, query, max_results):
            return [{
                "date": "2026-08-15T11:30:00+00:00",
                "title": "Current captain interview",
                "body": "The current team captain discussed the coming season.",
                "url": "https://example.com/current-captain",
                "source": "Example Sports",
            }]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)

    assert await web_search_mod._duckduckgo(
        "current team captain",
        [],
        count=3,
        topic="news",
    ) == [{
        "title": "Current captain interview",
        "url": "https://example.com/current-captain",
        "snippet": "The current team captain discussed the coming season.",
        "search_provider": "DuckDuckGo News",
        "published_date": "2026-08-15T11:30:00+00:00",
        "publisher": "Example Sports",
    }]


async def test_search_does_not_hide_non_plan_tavily_status_errors(monkeypatch):
    async def rejected(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "rate limited",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(429),
        )

    monkeypatch.setattr(web_search_mod, "_tavily", rejected)

    with pytest.raises(httpx.HTTPStatusError):
        await web_search_mod.tavily_search("query", ["nyc.gov"])


def _ctx():
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


async def _run(tool, query, **args):
    return await tool.handler({"query": query, **args}, _ctx())


async def test_web_search_ranks_by_tier_and_disclaims_community():
    async def fake_search(query, allowed, published_after=None, published_before=None, count=5):
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
    assert citations["S1"]["provenance"] == {
        "evidence_grade": "authoritative_excerpt",
        "source_tier": "authoritative",
    }
    assert citations["S2"]["provenance"] == {"evidence_grade": "discovery"}


async def test_web_search_keeps_ugc_off_allowlist_as_an_unverified_lead():
    async def fake_search(query, allowed, published_after=None, published_before=None, count=5):
        return [{"title": "Luma party", "url": "https://lu.ma/x", "snippet": "s"}]

    tool = web_search_tools(["nycgovparks.org"], source_tiers={}, search_fn=fake_search)[0]
    out = await _run(tool, "house party")
    assert "lu.ma" in out
    assert "search excerpt, cite only what it states" in out.lower()


async def test_web_search_prefer_boosts_domain():
    calls = []

    async def fake_search(
        query,
        allowed,
        published_after=None,
        published_before=None,
        count=5,
        include_domains=None,
    ):
        calls.append(include_domains)
        if include_domains:
            return [{"title": "WorldCup NYC", "url": "https://worldcup.nyc/b", "snippet": "s"}]
        return [
            {"title": "Time Out", "url": "https://timeout.com/a", "snippet": "s"},
        ]

    tiers = {"timeout.com": ("editorial", "events"), "worldcup.nyc": ("editorial", "world_cup")}
    tool = web_search_tools(["timeout.com", "worldcup.nyc"], source_tiers=tiers, search_fn=fake_search)[0]
    out = await _run(tool, "watch party", prefer=["worldcup.nyc"])
    assert out.index("worldcup.nyc") < out.index("timeout.com")  # preferred domain first
    assert calls == [None, ["worldcup.nyc"]]


async def test_web_search_does_not_repeat_when_preferred_domain_is_already_present():
    calls = 0

    async def fake_search(query, allowed, published_after=None, published_before=None, count=5, include_domains=None):
        nonlocal calls
        calls += 1
        return [{"title": "NYC", "url": "https://nyc.gov/a", "snippet": "s"}]

    tool = web_search_tools(["nyc.gov"], search_fn=fake_search)[0]
    await _run(tool, "parks", prefer=["nyc.gov"])
    assert calls == 1


# --- One search operation with optional recency and source grading ---

def _by_name(allow, tiers=None, news=None, search_fn=None):
    return {t.name: t for t in web_search_tools(allow, source_tiers=tiers, news_tier=news, search_fn=search_fn)}


def test_web_search_contract_explains_curated_excerpts_and_preferred_retrieval():
    tool = _by_name(["nyc.gov"])["web_search"]

    assert "curated editorial/news excerpts" in tool.description
    prefer = tool.parameters["properties"]["prefer"]["description"]
    assert "targeted search" in prefer
    assert "does not discard" in prefer


async def test_default_web_search_keeps_trust_domains_as_ranking_metadata():
    seen: dict[str, list[str]] = {}

    async def spy(query, domains, published_after=None, published_before=None, count=5):
        seen[query] = list(domains)
        return [{"title": "Gov", "url": "https://www.nyc.gov/x", "snippet": "s"}]

    tools = _by_name(["nyc.gov"], news=["gothamist.com", "nytimes.com"], search_fn=spy)
    await _run(tools["web_search"], "q")
    assert seen["q"] == ["gothamist.com", "nyc.gov", "nytimes.com"]


async def test_web_search_publication_bound_keeps_trust_domains_as_ranking_metadata():
    """A publication bound keeps the trust domains as ranking metadata."""
    seen: dict[str, set[str]] = {}

    async def spy(query, domains, published_after=None, count=5):
        assert published_after == "2026-08-01"
        seen[query] = set(domains)
        return []

    tools = _by_name(["nyc.gov"], news=["gothamist.com", "NYTimes.com"], search_fn=spy)
    out = await _run(tools["web_search"], "q", published_after="2026-08-01")
    assert {"nyc.gov", "gothamist.com", "nytimes.com"} <= seen["q"]  # allowlist ∪ news (lowercased)
    assert "no results" in out.lower()


async def test_web_search_ranks_news_below_gov():
    """A news-tier URL ranks below a gov URL, and carries the developing-news label."""
    async def fake(query, domains, published_after=None, count=5):
        assert published_after == "2025-08-12"
        return [
            {"title": "Gothamist story", "url": "https://gothamist.com/a", "snippet": "s"},
            {"title": "Official CCHR", "url": "https://www.nyc.gov/cchr", "snippet": "s"},
        ]

    tools = _by_name(["nyc.gov"], news=["gothamist.com"], search_fn=fake)
    out = await _run(
        tools["web_search"], "voucher ruling", published_after="2025-08-12"
    )
    assert out.index("nyc.gov") < out.index("gothamist.com")  # gov (authoritative) above news
    assert "📰 news" in out                                    # developing-news label present


async def test_web_search_grades_a_known_news_result_as_news():
    async def fake(query, domains, published_after=None, published_before=None, count=5):
        return [{"title": "Gothamist", "url": "https://gothamist.com/a", "snippet": "s"}]

    tools = _by_name(["nyc.gov"], news=["gothamist.com"], search_fn=fake)
    out = await _run(tools["web_search"], "q")
    assert "gothamist.com" in out
    assert "📰 news" in out


# --- Agent-settable publication bounds: what dates reach Tavily ---

def _capture_tavily(monkeypatch):
    """Patch the shared `_tavily` and record the date-bound kwargs each call passes."""
    calls: list[dict] = []

    async def fake_tavily(query, allowed_domains, **extra):
        calls.append(extra)
        return []

    monkeypatch.setattr(web_search_mod, "_tavily", fake_tavily)
    return calls


def _real_backend_tools():
    """Wire the production backend so optional publication bounds reach Tavily."""
    return {t.name: t for t in web_search_tools(["nyc.gov"])}


async def test_web_search_passes_publication_date_bounds_to_tavily(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["web_search"].handler(
        {
            "query": "q",
            "published_after": "2025-09-01",
            "published_before": "2025-12-01",
        },
        _ctx(),
    )
    assert calls[0]["start_date"] == "2025-09-01"
    assert calls[0]["end_date"] == "2025-12-01"


async def test_web_search_passes_news_topic_to_tavily(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()

    await tools["web_search"].handler(
        {"query": "current NYC sports news", "topic": "news"},
        _ctx(),
    )

    assert calls[0]["topic"] == "news"


async def test_web_search_rejects_invalid_or_reversed_publication_bounds():
    async def must_not_search(*_args, **_kwargs):
        raise AssertionError("invalid date bounds must not reach the backend")

    tool = web_search_tools(["nyc.gov"], search_fn=must_not_search)[0]

    invalid = await tool.handler(
        {"query": "q", "published_after": "last Tuesday"}, _ctx()
    )
    reversed_range = await tool.handler(
        {
            "query": "q",
            "published_after": "2025-12-01",
            "published_before": "2025-09-01",
        },
        _ctx(),
    )

    assert invalid == "Publication dates must use YYYY-MM-DD."
    assert reversed_range == "published_after must be earlier than published_before."


async def test_bounded_search_does_not_use_an_untimed_fallback(monkeypatch):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def must_not_fallback(*_args, **_kwargs):
        raise AssertionError("untimed fallback cannot honor exact publication bounds")

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", must_not_fallback)

    assert await web_search_mod.tavily_search(
        "query", ["nyc.gov"], published_after="2025-09-01"
    ) == []


async def test_topic_specific_search_marks_the_general_fallback_as_degraded(monkeypatch):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def fallback(*_args, **_kwargs):
        return [{
            "title": "Current reporting",
            "url": "https://example.com/current",
            "snippet": "Current lead",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", fallback)

    assert await web_search_mod.tavily_search(
        "current sports reporting", [], topic="news",
    ) == [{
        "title": "Current reporting",
        "url": "https://example.com/current",
        "snippet": "Current lead",
        "search_provider": "DuckDuckGo",
        "degraded_from_topic": "news",
    }]


async def test_web_search_without_publication_bounds_stays_untimed(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["web_search"].handler({"query": "q"}, _ctx())
    assert "start_date" not in calls[0]
    assert "end_date" not in calls[0]


async def test_web_search_passes_no_time_range(monkeypatch):
    """Regression guard: the untimed default web_search must never send a time_range."""
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()
    await tools["web_search"].handler({"query": "q"}, _ctx())
    assert "time_range" not in calls[0]


def test_web_search_exposes_publication_date_bounds():
    tools = web_search_tools(["nyc.gov"], news_tier=["gothamist.com"])
    assert [tool.name for tool in tools] == ["web_search"]
    properties = tools[0].parameters["properties"]
    assert properties["published_after"]["format"] == "date"
    assert properties["published_before"]["format"] == "date"
    assert "exclusive lower bound" in properties["published_after"]["description"]
    assert "exclusive upper bound" in properties["published_before"]["description"]
    assert "published_within" not in properties
    assert "recency" not in properties


def test_web_search_exposes_tavily_topic_without_changing_the_default():
    properties = web_search_tools(["nyc.gov"])[0].parameters["properties"]

    assert properties["topic"]["enum"] == ["general", "news", "finance"]
    assert "publication dates" in properties["topic"]["description"]


async def test_web_search_count_bounds_the_returned_results():
    seen = {}

    async def fake(query, domains, published_after=None, published_before=None, count=5):
        seen["count"] = count
        return [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": "evidence",
            }
            for index in range(5)
        ]

    tool = web_search_tools(["nyc.gov"], search_fn=fake)[0]
    assert tool.parameters["properties"]["count"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "description": "Maximum number of results to return, from 1 to 10.",
    }

    out = await tool.handler({"query": "NYC events", "count": 2}, _ctx())

    assert seen["count"] == 2
    assert "Result 0" in out
    assert "Result 1" in out
    assert "Result 2" not in out


async def test_handler_sends_the_model_query_without_hidden_rewriting():
    seen = []

    async def fake_search(query, domains, published_after=None, published_before=None, count=5):
        seen.append(query)
        return [{"title": "Official", "url": "https://www.nyc.gov/wc", "snippet": "row"}]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    out = await tool.handler(
        {"query": "what to prepare for tomorrow wc game July 18, 2026"}, ctx,
    )

    assert seen == ["what to prepare for tomorrow wc game July 18, 2026"]
    assert "Searched as" not in out


def test_web_search_stays_available_for_fresh_event_context():
    tools = {t.name: t for t in web_search_tools(["nyc.gov"])}
    desc = tools["web_search"].description.lower()
    assert "current" in desc
    assert "long-tail" in desc
    assert "events" in desc
    assert "always available" in desc


def test_web_search_description_does_not_duplicate_answer_policy():
    desc = web_search_tools(["nyc.gov"])[0].description.lower()
    assert "same missing fact" not in desc
    assert "final_answer" not in desc
    assert "say you could not confirm" not in desc


def test_web_search_parameter_descriptions_state_real_limits():
    properties = web_search_tools(["nyc.gov"])[0].parameters["properties"]
    assert "use `prefer` instead of `site:`" in properties["query"]["description"].lower()
    assert "publication or last-update date" in properties["published_after"]["description"]
    assert "not the date of an event" in properties["published_after"]["description"]
    assert "exclusive" in properties["published_before"]["description"]
    assert "targeted search" in properties["prefer"]["description"]
    assert "does not discard" in properties["prefer"]["description"]
