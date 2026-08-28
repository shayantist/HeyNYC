from __future__ import annotations

from datetime import datetime

import httpx
import pytest

import heynyc.core.tools.web_fetch as web_fetch_module
import heynyc.core.tools.web_search as web_search_module
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext, ToolFailure
from heynyc.core.tools.web_fetch import web_fetch_tools
from heynyc.core.tools.web_search import web_search_tools


def _context() -> ToolContext:
    return ToolContext(citations=CitationRegistry(), registry=Registry([]))


def test_shared_web_tools_expose_only_the_approved_contract() -> None:
    search = web_search_tools([])[0]
    fetch = web_fetch_tools()[0]

    assert set(search._input_schema()["properties"]) == {
        "queries",
        "domains",
        "published_after",
        "published_before",
    }
    assert set(fetch._input_schema()["properties"]) == {"url", "find"}
    assert fetch._input_schema()["properties"]["url"]["format"] == "uri"


@pytest.mark.asyncio
async def test_web_search_keeps_results_grouped_by_query() -> None:
    async def search(query, _known_domains, **_kwargs):
        slug = query.replace(" ", "-")
        return [{
            "title": query,
            "url": f"https://example.org/{slug}",
            "snippet": f"Evidence for {query}",
        }]

    result = await web_search_tools([], search_fn=search)[0].invoke(
        {"queries": ["student OMNY", "MTA authority"]},
        _context(),
    )

    assert "QUERY: student OMNY" in result
    assert "QUERY: MTA authority" in result


@pytest.mark.asyncio
async def test_web_search_returns_typed_failure_when_every_provider_is_unavailable() -> None:
    async def unavailable(*_args, **_kwargs):
        raise web_search_module.SearchUnavailable("providers unavailable")

    result = await web_search_tools([], search_fn=unavailable)[0].invoke(
        {"queries": ["current NYC service"]},
        _context(),
    )

    assert result == ToolFailure(
        status="unavailable",
        reason="Web search providers are unavailable: providers unavailable",
        retryable=True,
    )


@pytest.mark.asyncio
async def test_web_search_preserves_successful_query_groups_when_one_query_fails() -> None:
    async def mixed(query, _known_domains, **_kwargs):
        if query == "broken query":
            raise web_search_module.SearchUnavailable("provider failed")
        return [{
            "title": "Useful result",
            "url": "https://example.org/useful",
            "snippet": "Useful evidence",
        }]

    result = await web_search_tools([], search_fn=mixed)[0].invoke(
        {"queries": ["working query", "broken query"]},
        _context(),
    )

    assert "QUERY: working query" in result
    assert "Useful evidence" in result
    assert "QUERY FAILED: broken query" in result
    assert "provider failed" in result


@pytest.mark.asyncio
async def test_web_search_domains_exclude_results_from_other_hosts() -> None:
    provider_domains = []

    async def mixed_hosts(_query, _known_domains, **kwargs):
        provider_domains.append(kwargs.get("include_domains"))
        return [
            {
                "title": "Official result",
                "url": "https://www.nyc.gov/official",
                "snippet": "Official evidence",
            },
            {
                "title": "Other result",
                "url": "https://example.org/other",
                "snippet": "Other evidence",
            },
        ]

    result = await web_search_tools([], search_fn=mixed_hosts)[0].invoke(
        {"queries": ["NYC guidance"], "domains": [" NYC.GOV "]},
        _context(),
    )

    assert "https://www.nyc.gov/official" in result
    assert "https://example.org/other" not in result
    assert provider_domains == [["nyc.gov"]]


@pytest.mark.asyncio
async def test_fallback_results_identify_the_failed_provider(monkeypatch) -> None:
    async def tavily(*_args, **_kwargs):
        raise web_search_module.SearchUnavailable("Tavily unavailable")

    async def brave(*_args, **_kwargs):
        return [{"title": "Result", "url": "https://example.org", "snippet": "Evidence"}]

    monkeypatch.setattr(web_search_module, "_tavily", tavily)
    monkeypatch.setattr(web_search_module, "_brave", brave)

    results = await web_search_module._search_with_fallback("query", [])

    assert results[0]["search_provider"] == "Brave Web Search API"
    assert results[0]["degraded_providers"] == ["Tavily unavailable"]


@pytest.mark.asyncio
async def test_fallback_results_identify_empty_providers(monkeypatch) -> None:
    async def tavily(*_args, **_kwargs):
        return []

    async def brave(*_args, **_kwargs):
        return []

    async def duckduckgo(*_args, **_kwargs):
        return [{"title": "Result", "url": "https://example.org", "snippet": "Evidence"}]

    monkeypatch.setattr(web_search_module, "_tavily", tavily)
    monkeypatch.setattr(web_search_module, "_brave", brave)
    monkeypatch.setattr(web_search_module, "_duckduckgo", duckduckgo)

    results = await web_search_module._search_with_fallback("query", [])

    assert results[0]["search_provider"] == "DuckDuckGo"
    assert results[0]["degraded_providers"] == [
        "Tavily returned no results",
        "Brave returned no results",
    ]


@pytest.mark.asyncio
async def test_web_fetch_returns_numbered_text_from_find(monkeypatch) -> None:
    url = "https://example.org/page"
    page = web_fetch_module._FetchedPage(
        final_url=url,
        title="Page",
        text="Introduction\nStudent OMNY excludes express buses\nAuthority belongs to the MTA",
        acquisition=web_fetch_module.WebFetchAcquisition(
            requested_url=url,
            final_url=url,
            citation_url=url,
            route="http",
            fetched_at=datetime.now().astimezone(),
        ),
    )

    async def fetched(*_args, **_kwargs):
        return page

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fetched)
    tool = web_fetch_tools()[0]

    found = await tool.invoke({"url": url, "find": "express buses"}, _context())

    assert "L2: Student OMNY excludes express buses" in found
    assert "L1: Introduction" not in found


@pytest.mark.asyncio
async def test_web_fetch_returns_typed_failure_after_static_and_browser_fail(monkeypatch) -> None:
    url = "https://example.org/blocked"

    async def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", unavailable)

    result = await web_fetch_tools()[0].invoke({"url": url}, _context())

    assert result == ToolFailure(
        status="unavailable",
        reason="The page could not be retrieved by HTTP or browser.",
        retryable=False,
        source_url=url,
    )
