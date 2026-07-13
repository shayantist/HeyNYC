"""Offline-testable source content-drift detection prototype.

This module is intentionally not registered with the eval gate or imported by the
live agent. Callers supply the async fetch function so this module performs no
network activity unless an explicit consumer chooses to provide it.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class SourceBaseline:
    """The validators and normalized text snapshot captured for one source page."""

    url: str
    etag: str = ""
    last_modified: str = ""
    content_hash: str = ""


@dataclass
class DriftResult:
    """The result of comparing a fetched source page to its captured baseline."""

    url: str
    status: str
    detail: str = ""


Fetch = Callable[[str, dict[str, str]], Awaitable[object]]
_WHITESPACE_RE = re.compile(r"\s+")


def normalized_text_hash(text: str) -> str:
    """Return a SHA-256 hash after stripping and collapsing whitespace in *text*."""
    normalized = _WHITESPACE_RE.sub(" ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _header(headers: object, name: str) -> str:
    """Get a response header case-insensitively from a mapping-like object."""
    if not isinstance(headers, Mapping):
        return ""
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return ""


async def check_drift(baseline: SourceBaseline, fetch: Fetch) -> DriftResult:
    """Compare a source page with *baseline* using injected conditional fetching.

    The supplied fetcher receives the source URL and conditional headers, then
    returns an object with ``status_code``, ``headers``, and ``text`` attributes.
    Exceptions, missing responses, and explicit error responses are unreachable.
    """
    headers: dict[str, str] = {}
    if baseline.etag:
        headers["If-None-Match"] = baseline.etag
    if baseline.last_modified:
        headers["If-Modified-Since"] = baseline.last_modified

    try:
        response = await fetch(baseline.url, headers)
    except Exception as exc:
        return DriftResult(baseline.url, "unreachable", f"fetch failed: {exc}")

    status_code = getattr(response, "status_code", None)
    if response is None or getattr(response, "error", False) or not isinstance(status_code, int) or status_code <= 0:
        return DriftResult(baseline.url, "unreachable", "fetch returned a network error")

    if status_code == 304:
        return DriftResult(baseline.url, "unchanged", "validator: HTTP 304 Not Modified")

    response_etag = _header(getattr(response, "headers", {}), "ETag")
    if baseline.etag and response_etag:
        if response_etag == baseline.etag:
            return DriftResult(baseline.url, "unchanged", "validator: ETag matched")
        return DriftResult(baseline.url, "changed", "validator: ETag changed")

    if baseline.content_hash:
        current_hash = normalized_text_hash(str(getattr(response, "text", "")))
        if current_hash == baseline.content_hash:
            return DriftResult(baseline.url, "unchanged", "hash: normalized page text matched")
        return DriftResult(baseline.url, "changed", "hash: normalized page text changed")

    return DriftResult(
        baseline.url,
        "unchanged",
        "could not determine drift: no comparable baseline hash or validator",
    )


def baseline_from_citation(citation: dict[str, Any]) -> SourceBaseline:
    """Build a best-effort baseline from a citation's captured snippet and metadata."""
    snippet = str(citation.get("snippet", ""))
    return SourceBaseline(
        url=str(citation.get("url", "")),
        etag=str(citation.get("etag", "")),
        last_modified=str(citation.get("last_modified", "")),
        content_hash=normalized_text_hash(snippet),
    )
