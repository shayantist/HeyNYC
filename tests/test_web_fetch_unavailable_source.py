from __future__ import annotations

import pytest

import heynyc.core.tools.web_fetch as web_fetch_module
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


@pytest.mark.asyncio
async def test_failed_web_fetch_preserves_safe_url_as_unverified(monkeypatch):
    url = "https://otda.ny.gov/oah/"

    async def fail_fetch(*args, **kwargs):
        raise RuntimeError("unavailable")

    async def empty_search(_args, _ctx):
        return "No results from the live web search."

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fail_fetch)
    monkeypatch.setattr(
        web_fetch_module,
        "web_search_tools",
        lambda *_args, **_kwargs: [
            Tool(
                name="web_search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                handler=empty_search,
            )
        ],
    )
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler(
        {"url": url, "evidence_scope": "SNAP hearing request and deadline"},
        ctx,
    )

    assert url in result
    assert "unverified" in result.lower()
    assert "run one focused web_search" not in result
    assert "Focused search fallback" not in result
    assert "{cite:S1}" in result
    assert ctx.citations.mapping()["S1"] == {
        "id": "S1",
        "url": url,
        "title": "Unavailable source",
        "snippet": "No page content was retrieved.",
        "kind": "WEB",
        "valid_as_of": "",
        "provenance": {"evidence_grade": "unavailable"},
    }


@pytest.mark.asyncio
async def test_failed_web_fetch_runs_one_focused_search_fallback(monkeypatch):
    url = "https://otda.ny.gov/hearings/faq.asp"
    seen = []

    async def fail_fetch(*args, **kwargs):
        raise RuntimeError("unavailable")

    async def search(args, _ctx):
        seen.append(args)
        return "[S2] (authoritative) OTDA FAQ\nSNAP hearings may be requested within 90 days."

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fail_fetch)
    monkeypatch.setattr(
        web_fetch_module,
        "web_search_tools",
        lambda *_args, **_kwargs: [
            Tool(
                name="web_search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                handler=search,
            )
        ],
        raising=False,
    )
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler(
        {
            "url": url,
            "evidence_scope": "SNAP Fair Hearing request deadline",
        },
        ctx,
    )

    assert seen == [{
        "query": "SNAP Fair Hearing request deadline",
        "prefer": ["otda.ny.gov"],
        "count": 5,
    }]
    assert "The page could not be fetched" in result
    assert "SNAP hearings may be requested within 90 days" in result


@pytest.mark.asyncio
async def test_failed_web_fetch_does_not_echo_url_rejected_by_public_validation(monkeypatch):
    url = "https://rebinding.example/private"

    async def validate_then_fetch(candidate, *args, **kwargs):
        await web_fetch_module._validate_public_url(candidate)

    async def reject_private_url(*args, **kwargs):
        raise ValueError("private address")

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", validate_then_fetch)
    monkeypatch.setattr(web_fetch_module, "validate_and_resolve_url", reject_private_url)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler({"url": url}, ctx)

    assert result == "The page could not be fetched."
    assert url not in result


@pytest.mark.asyncio
async def test_clientless_fetch_validates_public_url_before_downloading(monkeypatch):
    url = "https://rebinding.example/private"
    downloaded = False

    async def reject_private_url(*args, **kwargs):
        raise ValueError("private address")

    async def record_download(*args, **kwargs):
        nonlocal downloaded
        downloaded = True
        raise AssertionError("unsafe URL reached downloader")

    monkeypatch.setattr(web_fetch_module, "validate_and_resolve_url", reject_private_url)
    monkeypatch.setattr(web_fetch_module, "safe_download", record_download)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler({"url": url}, ctx)

    assert result == "The page could not be fetched."
    assert url not in result
    assert downloaded is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/private",
        "https://[::1",
        "https://example.com:invalid/private",
    ],
)
async def test_failed_web_fetch_does_not_echo_malformed_or_credentialed_url(
    monkeypatch, url,
):
    async def fail_fetch(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fail_fetch)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler({"url": url}, ctx)

    assert result == "The page could not be fetched."
    assert url not in result
