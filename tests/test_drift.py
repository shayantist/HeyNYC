from __future__ import annotations

from types import SimpleNamespace

import pytest

from heynyc.eval.drift import (
    SourceBaseline,
    baseline_from_citation,
    check_drift,
    normalized_text_hash,
)


def _response(status_code: int, text: str = "", headers: dict[str, str] | None = None):
    return SimpleNamespace(status_code=status_code, text=text, headers=headers or {})


@pytest.mark.asyncio
async def test_304_response_is_unchanged():
    baseline = SourceBaseline(url="https://www.nyc.gov/page", etag='"v1"')

    async def fetch(url: str, headers: dict[str, str]):
        return _response(304)

    result = await check_drift(baseline, fetch)

    assert result.status == "unchanged"
    assert "validator" in result.detail


@pytest.mark.asyncio
async def test_matching_body_hash_is_unchanged():
    baseline = SourceBaseline(
        url="https://www.nyc.gov/page",
        content_hash=normalized_text_hash("Programs are available today."),
    )

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "Programs   are available\n today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "unchanged"
    assert "hash" in result.detail


@pytest.mark.asyncio
async def test_different_body_hash_is_changed():
    baseline = SourceBaseline(
        url="https://www.nyc.gov/page",
        content_hash=normalized_text_hash("Programs are available today."),
    )

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "Programs are closed today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "changed"
    assert "hash" in result.detail


@pytest.mark.asyncio
async def test_matching_etag_is_unchanged_even_when_body_differs():
    baseline = SourceBaseline(
        url="https://www.nyc.gov/page",
        etag='"v1"',
        content_hash=normalized_text_hash("Old body"),
    )

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "Different body", {"ETag": '"v1"'})

    result = await check_drift(baseline, fetch)

    assert result.status == "unchanged"
    assert "validator" in result.detail


@pytest.mark.asyncio
async def test_fetch_error_is_unreachable():
    baseline = SourceBaseline(url="https://www.nyc.gov/page")

    async def fetch(url: str, headers: dict[str, str]):
        raise OSError("connection refused")

    result = await check_drift(baseline, fetch)

    assert result.status == "unreachable"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [404, 500])
async def test_non_success_response_is_unreachable(status_code: int):
    baseline = SourceBaseline(url="https://www.nyc.gov/page", etag='"v1"')

    async def fetch(url: str, headers: dict[str, str]):
        return _response(status_code, "An error page", {"ETag": '"v1"'})

    result = await check_drift(baseline, fetch)

    assert result.status == "unreachable"
    assert str(status_code) in result.detail


@pytest.mark.asyncio
async def test_no_comparable_baseline_is_unknown():
    baseline = SourceBaseline(url="https://www.nyc.gov/page")

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "Programs are available today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "unknown"
    assert "could not be determined" in result.detail


@pytest.mark.asyncio
async def test_citation_snippet_probe_present_in_body_is_unchanged():
    baseline = baseline_from_citation({
        "url": "https://www.nyc.gov/page",
        "snippet": "Programs are available today.",
    })

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "The city confirms: Programs   are available\n today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "unchanged"
    assert "probe" in result.detail


@pytest.mark.asyncio
async def test_citation_snippet_probe_absent_from_body_is_changed():
    baseline = baseline_from_citation({
        "url": "https://www.nyc.gov/page",
        "snippet": "Programs are available today.",
    })

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "Programs are closed today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "changed"
    assert "probe" in result.detail


@pytest.mark.asyncio
async def test_direct_content_probe_ignores_whitespace_reflow():
    baseline = SourceBaseline(
        url="https://www.nyc.gov/page",
        content_probe="Programs   are available\n today.",
    )

    async def fetch(url: str, headers: dict[str, str]):
        return _response(200, "The city confirms: Programs are available today.")

    result = await check_drift(baseline, fetch)

    assert result.status == "unchanged"
    assert "probe" in result.detail


def test_normalized_text_hash_ignores_whitespace_reflow():
    assert normalized_text_hash("A service\n is   open") == normalized_text_hash(" A service is open ")


@pytest.mark.asyncio
async def test_conditional_headers_include_only_present_validators():
    captured: list[dict[str, str]] = []
    baseline = SourceBaseline(
        url="https://www.nyc.gov/page",
        etag='"v1"',
        last_modified="Mon, 13 Jul 2026 12:00:00 GMT",
    )

    async def fetch(url: str, headers: dict[str, str]):
        captured.append(headers)
        return _response(304)

    await check_drift(baseline, fetch)

    assert captured == [{
        "If-None-Match": '"v1"',
        "If-Modified-Since": "Mon, 13 Jul 2026 12:00:00 GMT",
    }]
