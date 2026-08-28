from __future__ import annotations

import httpx
import pytest

import heynyc.core.tools.web_fetch as web_fetch_module
from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext, ToolFailure


@pytest.mark.asyncio
async def test_failed_web_fetch_preserves_safe_url_as_unverified(monkeypatch):
    url = "https://otda.ny.gov/oah/"

    async def fail_fetch(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fail_fetch)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler(
        {"url": url},
        ctx,
    )

    assert result == ToolFailure(
        status="unavailable",
        reason="The page could not be retrieved by HTTP or browser.",
        retryable=False,
        source_url=url,
    )
    assert ctx.citations.mapping() == {}


@pytest.mark.asyncio
async def test_failed_web_fetch_returns_control_to_the_agent(monkeypatch):
    url = "https://otda.ny.gov/hearings/faq.asp"
    seen = []

    async def fail_fetch(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    async def search(args, _ctx):
        seen.append(args)
        raise AssertionError("web_fetch must not hide another search")

    monkeypatch.setattr(web_fetch_module, "_fetch_page_with_browser", fail_fetch)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await web_fetch_module.web_fetch_tools()[0].handler(
        {"url": url},
        ctx,
    )

    assert seen == []
    assert result.status == "unavailable"


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

    assert result.status == "rejected"
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

    assert result.status == "rejected"
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

    assert result.status == "rejected"
    assert url not in result
