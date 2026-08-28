from __future__ import annotations

import asyncio

import httpx
import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools import web_search as web_search_mod
from heynyc.core.tools.base import ToolContext
from heynyc.core.tools.web_search import _domain_allowed, web_search_tools

ALLOW = ["nyc.gov", "nyctourism.com", "nynjfwc26.com"]


def test_web_search_schema_accepts_a_flat_query_list():
    properties = web_search_tools(ALLOW)[0]._input_schema()["properties"]

    assert "queries" in properties
    assert "query" not in properties


async def test_web_search_runs_queries_concurrently_and_deduplicates_urls():
    started: set[str] = set()
    release = asyncio.Event()

    async def fake_search(query, domains, **kwargs):
        started.add(query)
        if len(started) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return [{
            "title": query,
            "url": "https://www.nyc.gov/shared",
            "snippet": f"Evidence for {query}",
        }]

    tool = web_search_tools(ALLOW, search_fn=fake_search)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    await tool.handler({"queries": ["student OMNY express bus", "who controls OMNY"]}, ctx)

    assert started == {"student OMNY express bus", "who controls OMNY"}
    assert len(ctx.citations) == 2


@pytest.fixture(autouse=True)
def _reset_tavily_plan_state(monkeypatch):
    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", False)


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
    snippet = "x" * 500 + " decisive detail"

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


async def test_web_search_reports_provider_failure_instead_of_claiming_no_results():
    async def unavailable(*_args, **_kwargs):
        raise web_search_mod.SearchUnavailable("Tavily, Brave, and DuckDuckGo failed")

    tool = web_search_tools(ALLOW, search_fn=unavailable)[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    out = await tool.handler({"query": "current NYC service"}, ctx)

    assert out.status == "unavailable"
    assert "providers" in out.reason.lower()


async def test_tavily_transport_failure_is_visible(monkeypatch):
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    with pytest.raises(web_search_mod.SearchUnavailable, match="ConnectError"):
        await web_search_mod._tavily("query", ["nyc.gov"])


async def test_tavily_valid_json_with_wrong_shape_degrades_to_no_results(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [None]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._tavily("query", []) == []


async def test_tavily_null_results_is_visible(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": None}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    with pytest.raises(web_search_mod.SearchUnavailable, match="invalid response"):
        await web_search_mod._tavily("query", [])


async def test_tavily_malformed_json_falls_through_to_brave(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return _Response()

    async def brave(*_args, **_kwargs):
        return [{
            "title": "Fallback",
            "url": "https://example.org/current",
            "snippet": "Current evidence",
            "search_provider": "Brave Web Search API",
        }]

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())
    monkeypatch.setattr(web_search_mod, "_brave", brave)
    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", False)

    results = await web_search_mod.tavily_search("query", [])

    assert results[0]["search_provider"] == "Brave Web Search API"


async def test_tavily_basic_search_does_not_restrict_results_to_known_domains(monkeypatch):
    request_json = {}
    request_headers = {}

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
            request_headers.update(kwargs["headers"])
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    await web_search_mod._tavily("things to do in NYC today", ["nyc.gov"])

    assert request_json["search_depth"] == "basic"
    assert "include_domains" not in request_json
    assert "api_key" not in request_json
    assert request_headers["Authorization"] == "Bearer configured"


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


async def test_tavily_does_not_promote_a_search_excerpt_to_page_content(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{
                "title": "Event guide",
                "url": "https://example.com/events",
                "content": "A provider-generated search excerpt.",
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

    result = await web_search_mod._tavily(
        "NYC events", [], search_depth="advanced", include_raw_content=True,
    )

    assert result[0]["snippet"] == "A provider-generated search excerpt."
    assert "raw_content" not in result[0]


async def test_web_search_can_return_query_focused_read_evidence(monkeypatch):
    calls = []

    async def fake_search(query, domains, **options):
        calls.append(options)
        return [{
            "title": "Weekly event guide",
            "url": "https://guide.example/this-week",
            "snippet": "A weekly guide.",
            "raw_content": "Accordion Festival is August 28 at Bryant Park.",
            "search_provider": "Tavily Search API",
        }]

    ctx = _ctx()
    monkeypatch.setattr(
        "heynyc.core.tools.web_fetch._validate_public_url",
        lambda _url: asyncio.sleep(0),
    )
    tool = web_search_tools(["guide.example"], search_fn=fake_search)[0]

    output = await tool.handler(
        {"query": "NYC events August 28", "include_page_evidence": True},
        ctx,
    )

    assert calls == [{
        "include_page_evidence": True,
    }]
    assert "Accordion Festival is August 28 at Bryant Park." in output
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
        "source_tier": "unverified",
        "search": {"provider": "Tavily Search API"},
        "acquisition_failure": "ValueError",
    }


async def test_web_search_batch_hydrates_selected_excerpt_pages(monkeypatch):
    async def fake_search(*_args, **_kwargs):
        return [{
            "title": "Weekly guide",
            "url": "https://guide.example/week",
            "snippet": "A current NYC event guide.",
            "search_provider": "DuckDuckGo",
        }]

    async def hydrate(result, _query, _ctx):
        return {
            **result,
            "raw_content": "Indie night is August 28 at Public Records.",
            "acquisition": {"route": "http"},
        }

    monkeypatch.setattr(web_search_mod, "_hydrate_search_result", hydrate)
    monkeypatch.setattr(
        "heynyc.core.tools.web_fetch._validate_public_url",
        lambda _url: asyncio.sleep(0),
    )
    ctx = _ctx()
    ctx.evidence_token_budget = 2_000
    tool = web_search_tools(["guide.example"], search_fn=fake_search)[0]

    output = await tool.handler(
        {
            "query": "indie music August 28",
            "include_page_evidence": True,
            "follow_relevant_links": True,
        }, ctx,
    )

    assert "Indie night is August 28 at Public Records" in output
    assert ctx.citations.mapping()["S1"]["provenance"]["acquisition"] == {"route": "http"}


async def test_web_search_does_not_follow_page_links_implicitly(monkeypatch):
    async def fake_search(*_args, **_kwargs):
        return [{
            "title": "Weekly guide",
            "url": "https://guide.example/week",
            "snippet": "A current NYC event guide.",
            "search_provider": "DuckDuckGo",
        }]

    async def hydrate(result, _query, _ctx):
        if result["url"].endswith("/week"):
            return {
                **result,
                "raw_content": "This week's guide.",
                "child_links": [
                    {"title": "Indie Night", "url": "https://venue.example/indie"},
                    {"title": "Restaurant directory", "url": "https://guide.example/food"},
                ],
                "acquisition": {"route": "http"},
            }
        return {
            **result,
            "raw_content": "Indie Night is August 28 at Public Records.",
            "child_links": [],
            "acquisition": {"route": "http"},
        }

    monkeypatch.setattr(web_search_mod, "_hydrate_search_result", hydrate)
    monkeypatch.setattr(
        "heynyc.core.tools.web_fetch._validate_public_url",
        lambda _url: asyncio.sleep(0),
    )
    ctx = _ctx()
    ctx.evidence_token_budget = 2_000
    tool = web_search_tools(["guide.example"], search_fn=fake_search)[0]

    output = await tool.handler(
        {
            "query": "indie music August 28",
            "include_page_evidence": True,
            "follow_relevant_links": True,
        }, ctx,
    )

    citations = ctx.citations.mapping()
    assert "Indie Night is August 28 at Public Records" not in output
    assert [citation["url"] for citation in citations.values()] == [
        "https://guide.example/week",
    ]


async def test_shared_hydration_preserves_child_links(monkeypatch):
    class Acquisition:
        def model_dump(self, **_kwargs):
            return {"route": "http"}

    class Page:
        text = "Weekly event guide"
        structured_events = []
        child_links = [{
            "url": "https://guide.example/events/indie-night",
            "title": "Indie Night",
        }]
        acquisition = Acquisition()

    async def fetch(_url, _client):
        return Page()

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fetch)

    result = await web_search_mod._hydrate_search_result(
        {"url": "https://guide.example/week", "snippet": "Weekly guide"},
        "indie night",
        _ctx(),
    )

    assert result["child_links"] == Page.child_links


@pytest.mark.asyncio
async def test_shared_hydration_exposes_page_read_failure(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fail)
    ctx = _ctx()
    result = await web_search_mod._hydrate_search_result(
        {"url": "https://example.com/event", "title": "Event", "snippet": "Excerpt"},
        "event time",
        ctx,
    )

    assert result["hydration_error"] == "ConnectError"
    snippet, provenance, _label = web_search_mod.search_result_evidence(result, ctx)
    assert "Page read failed" in snippet
    assert provenance["acquisition_failure"] == "ConnectError"


async def test_shared_hydration_preserves_complete_numbered_page_evidence(monkeypatch):
    class Acquisition:
        citation_url = "https://guide.example/week"

        def model_dump(self, **_kwargs):
            return {"route": "http"}

    class Page:
        text = (
            "Restaurant advertising and unrelated brunch recommendations. " * 200
            + "Indie concert August 28 at Public Records in Brooklyn."
        )
        structured_events = []
        child_links = []
        acquisition = Acquisition()

    async def fetch(_url, _client):
        return Page()

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fetch)

    result = await web_search_mod._hydrate_search_result(
        {"url": "https://guide.example/week", "snippet": "Weekly guide"},
        "indie concert August 28 Public Records",
        _ctx(),
    )

    assert "Indie concert August 28 at Public Records" in result["raw_content"]
    assert result["raw_content"].count("Restaurant advertising") == 200
    assert result["raw_content"].startswith("L1:")


async def test_shared_hydration_returns_generic_page_text(monkeypatch):
    class Acquisition:
        citation_url = "https://guide.example/week"

        def model_dump(self, **_kwargs):
            return {"route": "http", "citation_url": self.citation_url}

    class Page:
        text = "Weekly event guide"
        child_links = []
        acquisition = Acquisition()

    async def fetch(_url, _client):
        return Page()

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fetch)

    result = await web_search_mod._hydrate_search_result(
        {"url": "https://guide.example/week", "snippet": "Weekly guide"},
        "events",
        _ctx(),
    )

    assert result["raw_content"] == "L1: Weekly event guide"


async def test_provider_excerpts_are_batch_hydrated_as_generic_page_evidence(monkeypatch):
    class Acquisition:
        def model_dump(self, **_kwargs):
            return {"route": "http"}

    class Page:
        text = "Full event page"
        child_links = []
        acquisition = Acquisition()

    fetched = []

    async def fetch(url, _client):
        fetched.append(url)
        return Page()

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fetch)
    ctx = _ctx()
    ctx.evidence_token_budget = 100

    results = await web_search_mod.batch_hydrate_results(
        [{
            "url": "https://guide.example/event",
            "title": "Indie Night",
            "snippet": "Provider excerpt",
            "raw_content": "Provider excerpt",
        }],
        "indie night",
        ctx,
    )

    assert fetched == ["https://guide.example/event"]
    assert results[0]["raw_content"] == "L1: Full event page"


async def test_provider_read_context_is_not_fetched_again(monkeypatch):
    async def must_not_fetch(*_args):
        raise AssertionError("provider read context should not trigger a duplicate page read")

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", must_not_fetch)
    result = {
        "url": "https://guide.example/event",
        "title": "Indie Night",
        "snippet": "Provider context",
        "raw_content": "Provider context",
        "content_scope": "provider_context",
    }

    assert await web_search_mod.batch_hydrate_results(
        [result], "indie night", _ctx(),
    ) == [result]


async def test_provider_context_is_hydrated_when_required_fields_are_missing(monkeypatch):
    class Acquisition:
        def model_dump(self, **_kwargs):
            return {"route": "http"}

    class Page:
        text = "Indie Night is August 28 at Public Records."
        child_links = []
        acquisition = Acquisition()

    fetched = []

    async def fetch(url, _client):
        fetched.append(url)
        return Page()

    monkeypatch.setattr("heynyc.core.tools.web_fetch._fetch_page", fetch)
    result = {
        "url": "https://guide.example/week",
        "title": "Weekly guide",
        "snippet": "Provider context",
        "raw_content": "Provider context",
        "content_scope": "provider_context",
    }

    hydrated = await web_search_mod.hydrate_ranked_results(
        [result],
        "indie night",
        _ctx(),
        policy="fast",
        sufficient=lambda rows: "August 28" in rows[0].get("raw_content", ""),
        needs_hydration=lambda _result: True,
    )

    assert fetched == ["https://guide.example/week"]
    assert "Indie Night is August 28" in hydrated[0]["raw_content"]


def test_structured_but_incomplete_result_remains_hydratable():
    result = {
        "url": "https://guide.example/event",
        "title": "Incomplete event",
        "snippet": "Date missing",
        "structured_events": [{"name": "Incomplete event"}],
        "content_scope": "provider_context",
    }

    assert web_search_mod._hydration_candidates(
        [result], needs_hydration=lambda _result: True,
    ) == [0]


def test_web_search_schema_keeps_hydration_application_owned():
    properties = web_search_tools(["nyc.gov"])[0]._input_schema()["properties"]

    assert "include_page_evidence" not in properties
    assert "follow_relevant_links" not in properties


async def test_batch_hydration_does_not_spend_answer_context_capacity(monkeypatch):
    async def hydrate(result, _query, _ctx):
        return {
            **result,
            "raw_content": "internal extraction material " * 20,
            "acquisition": {"route": "http"},
        }

    monkeypatch.setattr(web_search_mod, "_hydrate_search_result", hydrate)
    monkeypatch.setattr(web_search_mod, "_text_tokens", lambda text, *_args: len(text))
    ctx = _ctx()
    ctx.evidence_token_budget = 100

    results = await web_search_mod.batch_hydrate_results([
        {"url": "https://guide.example/one", "title": "One", "snippet": "one"},
        {"url": "https://guide.example/two", "title": "Two", "snippet": "two"},
    ], "events", ctx)

    assert all("raw_content" in result for result in results)
    assert ctx.evidence_token_budget == 100


async def test_ranked_results_reject_non_https_provider_urls():
    async def fake_search(*_args, **_kwargs):
        return [
            {"url": "javascript:alert(1)", "title": "Unsafe", "snippet": "bad"},
            {"url": "https://guide.example/event", "title": "Safe", "snippet": "good"},
        ]

    results = await web_search_mod.retrieve_ranked_results(
        {"query": "NYC event"},
        _ctx(),
        search=fake_search,
    )

    assert [result["title"] for result in results] == ["Safe"]


async def test_ranked_results_drop_provider_urls_that_resolve_privately(monkeypatch):
    async def fake_search(*_args, **_kwargs):
        return [{
            "url": "https://internal.example/event",
            "title": "Unsafe",
            "snippet": "bad",
            "search_provider": "Test Search",
        }]

    async def reject(_url):
        raise ValueError("private address")

    monkeypatch.setattr("heynyc.core.tools.web_fetch._validate_public_url", reject)

    results = await web_search_mod.retrieve_ranked_results(
        {"query": "NYC event"},
        _ctx(),
        search=fake_search,
    )

    assert results == []


async def test_ranked_results_preserve_complete_numbered_provider_extracts():
    async def fake_search(*_args, **_kwargs):
        return [{
            "url": "https://guide.example/week",
            "title": "Weekly guide",
            "snippet": "Provider summary",
            "raw_content": (
                "Restaurant advertising and unrelated brunch recommendations. " * 200
                + "Indie concert August 28 at Public Records in Brooklyn."
            ),
            "content_scope": "provider_extract",
        }]

    results = await web_search_mod.retrieve_ranked_results(
        {"query": "indie concert August 28 Public Records"},
        _ctx(),
        search=fake_search,
    )

    assert "Indie concert August 28 at Public Records" in results[0]["raw_content"]
    assert results[0]["raw_content"].count("Restaurant advertising") == 200
    assert results[0]["raw_content"].startswith("L1:")


async def test_batch_hydration_relies_on_per_fetch_timeouts_and_runs_concurrently(monkeypatch):
    started = 0
    all_started = asyncio.Event()

    async def hydrate(result, _query, _ctx):
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        await all_started.wait()
        return {**result, "raw_content": "read"}

    monkeypatch.setattr(web_search_mod, "_hydrate_search_result", hydrate)

    result = await asyncio.wait_for(
        web_search_mod.batch_hydrate_results(
            [
                {"url": "https://guide.example/one", "title": "One", "snippet": "Lead"},
                {"url": "https://guide.example/two", "title": "Two", "snippet": "Lead"},
            ],
            "events",
            _ctx(),
        ),
        timeout=1,
    )

    assert all(result_item["raw_content"] == "read" for result_item in result)
    assert not hasattr(web_search_mod, "_BATCH_HYDRATION_TIMEOUT_S")


async def test_deep_hydration_processes_rank_order_and_cancels_after_sufficiency(monkeypatch):
    cancelled = asyncio.Event()

    async def hydrate(result, _query, _ctx):
        if result["title"] == "First":
            return {**result, "structured_events": [{"name": "Enough"}]}
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(web_search_mod, "_hydrate_search_result", hydrate)
    results = await web_search_mod.hydrate_ranked_results(
        [
            {"url": "https://example.com/first", "title": "First", "snippet": "lead"},
            {"url": "https://example.com/second", "title": "Second", "snippet": "lead"},
        ],
        "events",
        _ctx(),
        policy="deep",
        sufficient=lambda rows: any(row.get("structured_events") for row in rows),
    )

    assert results[0]["structured_events"] == [{"name": "Enough"}]
    assert cancelled.is_set()


async def test_web_search_keeps_tavily_metadata_internal_and_uses_score_as_a_tiebreaker():
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
    assert "Provider relevance score" not in out
    assert "Search provider:" not in out
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

    assert "Search provider:" not in out
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


async def test_search_falls_back_to_brave_when_tavily_plan_is_exhausted(monkeypatch):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def brave(query, allowed_domains, count=5, topic=None, **_options):
        assert query == "current NYC service"
        assert allowed_domains == ["nyc.gov"]
        assert count == 7
        assert topic is None
        return [{
            "title": "Official",
            "url": "https://nyc.gov/service",
            "snippet": "Current",
            "search_provider": "Brave Search API",
            "degraded_providers": ["plan limit exceeded"],
        }]

    async def duckduckgo(*_args, **_kwargs):
        return []

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_brave", brave)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", duckduckgo)

    assert await web_search_mod.tavily_search(
        "current NYC service", ["nyc.gov"], count=7
    ) == [{
        "title": "Official",
        "url": "https://nyc.gov/service",
        "snippet": "Current",
        "search_provider": "Brave Search API",
        "degraded_providers": ["plan limit exceeded"],
    }]


async def test_search_falls_back_to_brave_when_tavily_transport_fails(monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("provider unavailable")

    async def brave(*_args, **_kwargs):
        return [{
            "title": "Official",
            "url": "https://nyc.gov/service",
            "snippet": "Current",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily", unavailable)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    results = await web_search_mod.tavily_search("current NYC service", ["nyc.gov"])

    assert results[0]["search_provider"] == "Brave Web Search API"


async def test_search_falls_back_to_brave_when_tavily_is_not_configured(monkeypatch):
    async def brave(*_args, **_kwargs):
        return [{
            "title": "Official",
            "url": "https://nyc.gov/service",
            "snippet": "Current",
            "search_provider": "Brave Web Search API",
        }]

    monkeypatch.setattr(web_search_mod.config, "TAVILY_API_KEY", "")
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    results = await web_search_mod.tavily_search("current NYC service", ["nyc.gov"])

    assert results[0]["search_provider"] == "Brave Web Search API"


async def test_search_uses_duckduckgo_when_tavily_and_brave_are_unavailable(monkeypatch):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def unavailable(*_args, **_kwargs):
        return []

    async def fallback(*_args, **_kwargs):
        return [{"title": "Lead", "url": "https://example.com", "snippet": "Lead"}]

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_brave", unavailable)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", fallback)

    assert (await web_search_mod.tavily_search("query", []))[0]["search_provider"] == "DuckDuckGo"


async def test_secondary_search_stops_after_brave_returns_results(monkeypatch):
    calls = []

    async def brave(*_args, **_kwargs):
        calls.append("brave")
        return [{
            "title": "Editorial guide",
            "url": "https://guide.example/week",
            "snippet": "A weekly guide.",
            "search_provider": "Brave Web Search API",
        }]

    async def duckduckgo(*_args, **_kwargs):
        calls.append("duckduckgo")
        return [{
            "title": "Community listing",
            "url": "https://events.example/show",
            "snippet": "A direct event listing.",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", True)
    monkeypatch.setattr(web_search_mod, "_brave", brave)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", duckduckgo)

    results = await web_search_mod.tavily_search(
        "NYC events this week", [], include_page_evidence=True,
    )

    assert calls == ["brave"]
    assert [result["url"] for result in results] == ["https://guide.example/week"]


async def test_secondary_search_uses_duckduckgo_only_when_brave_is_empty(monkeypatch):
    async def brave(*_args, **_kwargs):
        return []

    async def duckduckgo(*_args, **_kwargs):
        return [{"title": "Listing", "url": "https://events.example", "snippet": "Listing"}]

    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", True)
    monkeypatch.setattr(web_search_mod, "_brave", brave)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", duckduckgo)

    results = await web_search_mod.tavily_search("NYC services", [])

    assert [result["url"] for result in results] == ["https://events.example"]


async def test_preferred_tavily_uses_one_focused_request_then_falls_back(monkeypatch):
    calls = []

    async def tavily(*_args, **kwargs):
        calls.append(("tavily", kwargs.get("include_domains")))
        raise httpx.ConnectError("focused request unavailable")

    async def brave(*_args, **kwargs):
        calls.append(("brave", kwargs.get("include_domains")))
        return [{"title": "Fallback", "url": "https://fallback.example", "snippet": "Fallback"}]

    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", False)
    monkeypatch.setattr(web_search_mod, "_tavily", tavily)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    results = await web_search_mod.tavily_search(
        "NYC events", [], include_domains=["preferred.example"],
    )

    assert [result["url"] for result in results] == ["https://fallback.example"]
    assert calls == [
        ("tavily", ["preferred.example"]),
        ("brave", ["preferred.example"]),
    ]


async def test_preferred_tavily_latches_plan_exhaustion_from_one_lane(monkeypatch):
    tavily_calls = 0

    async def tavily(*_args, **kwargs):
        nonlocal tavily_calls
        tavily_calls += 1
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def brave(*_args, **_kwargs):
        return [{"title": "Fallback", "url": "https://fallback.example", "snippet": "Fallback"}]

    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", False)
    monkeypatch.setattr(web_search_mod, "_tavily", tavily)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    first = await web_search_mod.tavily_search(
        "NYC events", [], include_domains=["preferred.example"],
    )
    second = await web_search_mod.tavily_search("NYC events", [])

    assert [result["url"] for result in first] == ["https://fallback.example"]
    assert [result["url"] for result in second] == ["https://fallback.example"]
    assert tavily_calls == 1


async def test_tavily_plan_exhaustion_skips_tavily_for_later_searches(monkeypatch):
    tavily_calls = 0

    async def exhausted(*_args, **_kwargs):
        nonlocal tavily_calls
        tavily_calls += 1
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def brave(*_args, **_kwargs):
        return [{
            "title": "Lead",
            "url": "https://example.com",
            "snippet": "Lead",
            "search_provider": "Brave Search API",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    await web_search_mod.tavily_search("first", [])
    await web_search_mod.tavily_search("second", [])

    assert tavily_calls == 1


async def test_brave_web_search_returns_all_ranked_results_and_extra_snippets(monkeypatch):
    request_params = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "web": {
                    "results": [{
                        "url": "https://example.com/event",
                        "title": "Event",
                        "description": "Event starts at 7 PM.",
                        "extra_snippets": [
                            "The venue is in Brooklyn.",
                        ],
                        "page_age": "2026-08-25T10:00:00Z",
                        "profile": {"long_name": "Example Events"},
                    }],
                },
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            assert url.endswith("/web/search")
            request_params.update(kwargs["params"])
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "BRAVE_SEARCH_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._brave(
        "NYC events", [], count=4, include_raw_content=True,
    ) == [{
        "title": "Event",
        "url": "https://example.com/event",
        "snippet": "Event starts at 7 PM.\n\nThe venue is in Brooklyn.",
        "search_provider": "Brave Web Search API",
        "page_age": "2026-08-25T10:00:00Z",
        "publisher": "Example Events",
    }]
    assert request_params == {
        "q": "NYC events",
        "count": 4,
        "extra_snippets": True,
    }


async def test_brave_malformed_json_falls_through_to_duckduckgo(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, **_kwargs):
            return _Response()

    async def duckduckgo(*_args, **_kwargs):
        return [{"title": "Fallback", "url": "https://example.org", "snippet": "ok"}]

    monkeypatch.setattr(web_search_mod.config, "BRAVE_SEARCH_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())
    monkeypatch.setattr(web_search_mod, "_duckduckgo", duckduckgo)
    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", True)

    results = await web_search_mod.tavily_search("query", [])

    assert results[0]["search_provider"] == "DuckDuckGo"


async def test_brave_valid_json_with_wrong_shape_returns_no_results(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"web": {"results": [None]}}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "BRAVE_SEARCH_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._brave("query", []) == []


async def test_brave_preserves_url_only_result(monkeypatch):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "web": {
                    "results": [{
                        "url": "https://example.org/source",
                        "title": "Possible source",
                    }]
                }
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "BRAVE_SEARCH_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    assert await web_search_mod._brave("query", []) == [{
        "title": "Possible source",
        "url": "https://example.org/source",
        "snippet": "",
        "search_provider": "Brave Web Search API",
    }]


async def test_brave_uses_native_reranking_for_preferred_domains(monkeypatch):
    request_params = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"web": {"results": []}}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, **kwargs):
            request_params.update(kwargs["params"])
            return _Response()

    monkeypatch.setattr(web_search_mod.config, "BRAVE_SEARCH_API_KEY", "configured")
    monkeypatch.setattr(web_search_mod.httpx, "AsyncClient", lambda **_kwargs: _Client())

    await web_search_mod._brave(
        "music and arts this week in NYC",
        [],
        include_domains=["donyc.com", "lu.ma", "partiful.com"],
    )

    assert request_params["q"] == "music and arts this week in NYC"
    assert request_params["goggles"] == "\n".join((
        "$boost,site=donyc.com",
        "$boost,site=lu.ma",
        "$boost,site=partiful.com",
    ))


async def test_duckduckgo_news_fallback_preserves_publication_metadata(monkeypatch):
    import ddgs

    backend = None

    class FakeDDGS:
        def news(self, query, max_results, **kwargs):
            nonlocal backend
            backend = kwargs.get("backend")
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
    assert backend == "duckduckgo"


async def test_duckduckgo_none_result_degrades_to_no_results(monkeypatch):
    import ddgs

    class FakeDDGS:
        def text(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)

    assert await web_search_mod._duckduckgo("query", []) == []


async def test_search_falls_back_after_non_plan_tavily_status_errors(monkeypatch):
    async def rejected(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "rate limited",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(429),
        )

    async def brave(*_args, **_kwargs):
        return [{
            "title": "Fallback",
            "url": "https://example.com/current",
            "snippet": "Current",
            "search_provider": "Brave Web Search API",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily", rejected)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    results = await web_search_mod.tavily_search("query", ["nyc.gov"])

    assert results[0]["search_provider"] == "Brave Web Search API"


async def test_brave_fallback_marks_lost_topic_scope(monkeypatch):
    async def brave(*_args, **_kwargs):
        return [{
            "title": "Current reporting",
            "url": "https://example.com/current",
            "snippet": "Current",
            "search_provider": "Brave Web Search API",
        }]

    monkeypatch.setattr(web_search_mod, "_tavily_plan_exhausted", True)
    monkeypatch.setattr(web_search_mod, "_brave", brave)

    results = await web_search_mod.tavily_search("current reporting", [], topic="news")

    assert results[0]["degraded_from_topic"] == "news"


def _ctx():
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


def test_provider_context_remains_excerpt_evidence():
    result = {
        "url": "https://example.com/current",
        "title": "Current listing",
        "snippet": "The event starts at 7 PM.",
        "raw_content": "The event starts at 7 PM.",
        "content_scope": "provider_context",
        "source_tier": "editorial",
        "search_provider": "Brave LLM Context API",
    }

    _snippet, provenance, _label = web_search_mod.search_result_evidence(result, _ctx())

    assert provenance["evidence_grade"] == "search_excerpt"


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
        return [
            {"title": "WorldCup NYC", "url": "https://worldcup.nyc/b", "snippet": "s"},
            {"title": "Time Out", "url": "https://timeout.com/a", "snippet": "s"},
        ]

    tiers = {"timeout.com": ("editorial", "events"), "worldcup.nyc": ("editorial", "world_cup")}
    tool = web_search_tools(["timeout.com", "worldcup.nyc"], source_tiers=tiers, search_fn=fake_search)[0]
    out = await _run(tool, "watch party", prefer=["worldcup.nyc"])
    assert out.index("worldcup.nyc") < out.index("timeout.com")  # preferred domain first
    assert calls == [None]


async def test_web_search_fuses_broad_and_focused_results_even_when_one_preferred_result_exists():
    calls = []

    async def fake_search(query, allowed, published_after=None, published_before=None, count=5, include_domains=None):
        calls.append(include_domains)
        return [
            {"title": "NYC Parks", "url": "https://nyc.gov/a", "snippet": "s"},
            {"title": "Open-web venue", "url": "https://venue.example/c", "snippet": "s"},
            {"title": "DoNYC concert", "url": "https://donyc.com/a", "snippet": "s"},
            {"title": "Luma show", "url": "https://luma.com/b", "snippet": "s"},
        ]

    tiers = {
        "nyc.gov": ("authoritative", "events"),
        "donyc.com": ("editorial", "events"),
        "luma.com": ("community", "events"),
    }
    tool = web_search_tools(list(tiers), source_tiers=tiers, search_fn=fake_search)[0]
    out = await tool.handler(
        {"query": "NYC music August 20, 2026", "prefer": ["donyc.com", "luma.com"]},
        ToolContext(
            citations=CitationRegistry(),
            registry=Registry([]),
            event_turn="discovery",
        ),
    )

    assert calls == [None]
    assert "NYC Parks" in out
    assert "Open-web venue" in out
    assert "DoNYC concert" in out
    assert "Luma show" in out


async def test_event_discovery_preserves_provider_relevance_over_source_tier():
    async def fake_search(*_args, **_kwargs):
        return [
            {
                "title": "Provider-ranked concert guide",
                "url": "https://guide.example/concerts",
                "snippet": "Concerts matching the request.",
            },
            {
                "title": "Lower-ranked city calendar",
                "url": "https://nyc.gov/events",
                "snippet": "A broad event calendar.",
            },
        ]

    tool = web_search_tools(
        ["nyc.gov"],
        source_tiers={"nyc.gov": ("authoritative", "events")},
        search_fn=fake_search,
    )[0]
    output = await tool.handler(
        {"query": "NYC concerts this week"},
        ToolContext(
            citations=CitationRegistry(),
            registry=Registry([]),
            event_turn="discovery",
        ),
    )

    assert output.index("Provider-ranked concert guide") < output.index(
        "Lower-ranked city calendar"
    )


async def test_web_search_merges_distinct_evidence_for_the_same_rank_fused_page():
    async def fake_search(
        query,
        allowed,
        published_after=None,
        published_before=None,
        count=5,
        include_domains=None,
    ):
        return [{
            "title": "Weekly guide",
            "url": "https://guide.example/week",
            "snippet": (
                "Friday has an accordion festival. Saturday has an outdoor concert."
            ),
        }]

    ctx = _ctx()
    tool = web_search_tools(["guide.example"], search_fn=fake_search)[0]

    await tool.handler(
        {"query": "NYC events this week", "prefer": ["guide.example"]},
        ctx,
    )

    snippet = ctx.citations.mapping()["S1"]["snippet"]
    assert "Friday has an accordion festival" in snippet
    assert "Saturday has an outdoor concert" in snippet


# --- One search operation with optional recency and source grading ---

def _by_name(allow, tiers=None, news=None, search_fn=None):
    return {t.name: t for t in web_search_tools(allow, source_tiers=tiers, news_tier=news, search_fn=search_fn)}


def test_web_search_contract_explains_ranked_excerpts():
    tool = _by_name(["nyc.gov"])["web_search"]

    assert "curated editorial/news excerpts" in tool.description
    assert "short noun-phrase query" in tool.description
    assert "ranked source excerpts" in tool.description
    assert "search again" not in tool.description
    assert "prefer" not in tool._input_schema()["properties"]


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


async def test_web_search_passes_shared_read_mode_to_production_backend(monkeypatch):
    calls = _capture_tavily(monkeypatch)
    tools = _real_backend_tools()

    await tools["web_search"].handler(
        {"query": "NYC events", "include_page_evidence": True},
        _ctx(),
    )

    assert calls[0]["search_depth"] == "advanced"
    assert "chunks_per_source" not in calls[0]
    assert calls[0]["include_raw_content"] is True


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


async def test_bounded_search_uses_last_resort_fallback_with_an_explicit_limitation(
    monkeypatch,
):
    async def exhausted(*_args, **_kwargs):
        raise httpx.HTTPStatusError(
            "plan limit exceeded",
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(432),
        )

    async def fallback(*_args, **_kwargs):
        return [{
            "title": "Possible match",
            "url": "https://example.com/match",
            "snippet": "Possible current match",
        }]

    async def no_brave(*_args, **_kwargs):
        return []

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_brave", no_brave)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", fallback)

    assert await web_search_mod.tavily_search(
        "query", ["nyc.gov"], published_after="2025-09-01"
    ) == [{
        "title": "Possible match",
        "url": "https://example.com/match",
        "snippet": "Possible current match",
            "search_provider": "DuckDuckGo",
            "degraded_providers": ["plan limit exceeded", "Brave returned no results"],
            "degraded_publication_bounds": {
            "published_after": "2025-09-01",
            "published_before": None,
        },
    }]


async def test_degraded_bounded_fallback_cannot_become_answer_grade():
    result = {
        "title": "Official page with unknown freshness",
        "url": "https://www.nyc.gov/site/example/page.page",
        "snippet": "The page states an eligibility rule.",
        "source_tier": "authoritative",
        "degraded_publication_bounds": {
            "published_after": "2026-08-01",
            "published_before": None,
        },
    }

    _snippet, provenance, _label = web_search_mod.search_result_evidence(
        result, _ctx(),
    )

    assert provenance["evidence_grade"] == "discovery"
    assert provenance["source_tier"] == "unverified"


async def test_sparse_search_result_preserves_its_url_as_a_discovery_lead():
    async def search(*_args, **_kwargs):
        return [{
            "title": "Possible official source",
            "url": "https://www.nyc.gov/site/example/page.page",
            "snippet": "",
        }]

    ctx = _ctx()
    output = await web_search_tools([], search_fn=search)[0].handler(
        {"query": "possible official source"}, ctx,
    )

    assert "No page content was retrieved" in output
    assert "https://www.nyc.gov/site/example/page.page" in output
    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
        "source_tier": "unverified",
    }


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

    async def no_brave(*_args, **_kwargs):
        return []

    monkeypatch.setattr(web_search_mod, "_tavily", exhausted)
    monkeypatch.setattr(web_search_mod, "_brave", no_brave)
    monkeypatch.setattr(web_search_mod, "_duckduckgo", fallback)

    assert await web_search_mod.tavily_search(
        "current sports reporting", [], topic="news",
    ) == [{
        "title": "Current reporting",
        "url": "https://example.com/current",
        "snippet": "Current lead",
            "search_provider": "DuckDuckGo",
            "degraded_providers": ["plan limit exceeded", "Brave returned no results"],
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
    properties = tools[0]._input_schema()["properties"]
    assert properties["published_after"]["anyOf"][0]["format"] == "date"
    assert properties["published_before"]["anyOf"][0]["format"] == "date"
    assert "published_within" not in properties
    assert "recency" not in properties


def test_web_search_exposes_tavily_topic_without_changing_the_default():
    properties = web_search_tools(["nyc.gov"])[0]._input_schema()["properties"]

    assert "topic" not in properties


def test_web_search_keeps_shared_page_evidence_mode_internal():
    properties = web_search_tools(["nyc.gov"])[0]._input_schema()["properties"]

    assert "include_page_evidence" not in properties
    assert "follow_relevant_links" not in properties


async def test_web_search_preserves_all_provider_results_without_a_model_count():
    async def fake(query, domains, published_after=None, published_before=None):
        return [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": "evidence",
            }
            for index in range(5)
        ]

    tool = web_search_tools(["nyc.gov"], search_fn=fake)[0]
    assert "count" not in tool._input_schema()["properties"]

    out = await tool.handler({"query": "NYC events"}, _ctx())

    assert "Result 0" in out
    assert "Result 1" in out
    assert "Result 2" in out
    assert "Result 4" in out


async def test_web_search_packs_ranked_evidence_only_when_model_budget_requires_it(monkeypatch):
    async def fake(query, domains, published_after=None, published_before=None):
        return [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": "evidence " * 20,
            }
            for index in range(3)
        ]

    monkeypatch.setattr(
        "heynyc.core.tools.web_search._text_tokens",
        lambda text, _model=None: len(text.split()),
    )
    ctx = _ctx()
    ctx.evidence_token_budget = 60

    out = await web_search_tools(["nyc.gov"], search_fn=fake)[0].handler(
        {"query": "NYC events"}, ctx
    )

    assert "Result 0" in out
    assert "Result 2" not in out
    assert "returned 3 ranked pages" in out
    assert "Refine the query" not in out
    assert len(ctx.citations.mapping()) < 3
    assert 0 <= ctx.evidence_token_budget < 60


async def test_exhausted_evidence_budget_retains_highest_ranked_source_url(monkeypatch):
    async def fake(*_args, **_kwargs):
        return [{
            "title": "Highest-ranked source",
            "url": "https://example.com/source",
            "snippet": "Useful evidence that cannot fit.",
        }]

    monkeypatch.setattr(
        "heynyc.core.tools.web_search._text_tokens",
        lambda text, _model=None: len(text.split()),
    )
    ctx = _ctx()
    ctx.evidence_token_budget = 0

    out = await web_search_tools([], search_fn=fake)[0].handler(
        {"query": "NYC events"}, ctx,
    )

    assert "https://example.com/source" in out
    assert "No evidence fit" in out
    assert ctx.citations.mapping()["S1"]["provenance"]["evidence_grade"] == "discovery"


async def test_web_search_uses_provider_evidence_without_local_passage_ranking(monkeypatch):
    async def fake_search(*_args, **_kwargs):
        return [{
            "title": "Large event guide",
            "url": "https://guide.example/week",
            "snippet": "Guide summary",
            "raw_content": "full page",
        }]

    monkeypatch.setattr(
        web_search_mod,
        "_text_tokens",
        lambda text, _model=None: len(text.split()),
    )
    ctx = _ctx()
    ctx.evidence_token_budget = 35

    output = await web_search_tools(["guide.example"], search_fn=fake_search)[0].handler(
        {"query": "indie concert August 28"}, ctx,
    )

    assert "full page" in output

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


def test_web_search_description_does_not_duplicate_answer_policy():
    desc = web_search_tools(["nyc.gov"])[0].description.lower()
    assert "same missing fact" not in desc
    assert "final_answer" not in desc
    assert "say you could not confirm" not in desc


def test_web_search_parameter_descriptions_state_real_limits():
    properties = web_search_tools(["nyc.gov"])[0]._input_schema()["properties"]
    assert properties["queries"]["description"] == "Independent focused searches"
    assert properties["published_after"]["description"] == "Publication date lower bound"
    assert properties["published_before"]["description"] == "Publication date upper bound"
