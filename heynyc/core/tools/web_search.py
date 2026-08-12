"""Web search for fresh and long-tail information.

Known domains are trust metadata, not a retrieval filter. The search backend is
injectable for tests; production uses Tavily Basic and marks unknown sources as
unverified leads rather than silently discarding them.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from .. import config
from ..registry import TIER_RANK
from .base import Tool, ToolContext

# (query, allowed_domains, published_after=None, published_before=None, count=5)
# -> result dictionaries
SearchFn = Callable[..., Awaitable[list[dict]]]

_ARCHIVE_WARNING = (
    "SOURCE STATUS: ARCHIVED. The publisher identifies this as historical or "
    "out-of-date content; do not present it as current."
)
_ARCHIVE_MARKERS = (
    "archived content",
    "out of date",
    "no longer current",
    "historical content",
)


def _domain_allowed(url: str, allowlist: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowlist)


def archive_warning(url: str, text: str = "") -> str:
    path = (urlparse(url).path or "").lower()
    haystack = text.lower()
    if "/archive/" in path or "save-policy-news-archive" in path:
        return _ARCHIVE_WARNING
    return _ARCHIVE_WARNING if any(marker in haystack for marker in _ARCHIVE_MARKERS) else ""


async def _tavily(query: str, allowed_domains: list[str], **extra) -> list[dict]:
    """Shared Tavily call. Returns [] when no API key (caller treats as unavailable)."""
    if not config.TAVILY_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": config.TAVILY_API_KEY,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "basic",
                    **extra,
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
    except (httpx.TimeoutException, httpx.TransportError):
        return []
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in results]


async def _duckduckgo(
    query: str,
    allowed_domains: list[str],
    count: int = 5,
) -> list[dict]:
    """Local search fallback for a Tavily plan-limit response."""
    from ddgs import DDGS

    try:
        results = await asyncio.to_thread(
            DDGS().text,
            query,
            max_results=count,
        )
    except Exception:
        return []
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", ""),
        }
        for result in results
        if result.get("href")
    ][:count]


async def _search_with_fallback(
    query: str,
    allowed_domains: list[str],
    *,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    count: int = 5,
    **tavily_options,
) -> list[dict]:
    try:
        return await _tavily(
            query, allowed_domains, max_results=count, **tavily_options
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 432:
            raise
        if published_after or published_before:
            return []
        return await _duckduckgo(
            query,
            allowed_domains,
            count=count,
        )


async def tavily_search(
    query: str,
    allowed_domains: list[str],
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    count: int = 5,
) -> list[dict]:
    """Tavily Basic search with optional publication-date bounds."""
    options = {}
    if published_after:
        options["start_date"] = published_after
    if published_before:
        options["end_date"] = published_before
    return await _search_with_fallback(
        query,
        allowed_domains,
        published_after=published_after,
        published_before=published_before,
        count=count,
        **options,
    )


_BASE_GOV = {"nyc.gov", "cityofnewyork.us", "mta.info"}


def _tier_of(
    url: str,
    source_tiers: dict[str, tuple[str, str]],
    news_tier: tuple[str, ...] | list[str] = (),
) -> str:
    """Best tier for a URL's host: an explicit source_tiers match (highest wins), else a default,
    gov domains are authoritative, a curated news-tier domain is `news` (subordinate), and
    everything else is unverified. Gov always outranks news so an official page can
    never be demoted by also appearing in the recency check."""
    host = (urlparse(url).hostname or "").lower()
    best: Optional[str] = None
    for domain, (tier, _module) in source_tiers.items():
        if host == domain or host.endswith("." + domain):
            if best is None or TIER_RANK.get(tier, 0) > TIER_RANK.get(best, 0):
                best = tier
    if best is not None:
        return best
    if host.endswith(".gov") or any(host == d or host.endswith("." + d) for d in _BASE_GOV):
        return "authoritative"
    if any(host == d.lower() or host.endswith("." + d.lower()) for d in news_tier):
        return "news"
    return "unverified"


def _prefers(url: str, prefer: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in prefer)


# Per-tier presentation label for a result block (falls back to the bare tier name).
_TIER_LABELS = {
    "community": "⚠️ community-posted, confirm before you go",
    "news": "📰 news, recent/developing, verify against the official source",
    "unverified": "⚠️ unverified source, check before relying on it",
}


def _make_handler(
    search: SearchFn,
    domains: list[str],
    source_tiers: dict[str, tuple[str, str]],
    news_tier: list[str],
    *,
    abstain_msg: str,
) -> Callable:
    """Build a search handler over `domains`, tagging + ranking results by trust tier."""
    async def _handler(args: dict, ctx: ToolContext) -> str:
        prefer = args.get("prefer") or []
        query = str(args["query"]).strip()
        count = args.get("count", 5)
        count = count if isinstance(count, int) and not isinstance(count, bool) else 5
        count = max(1, min(count, 10))
        published_after = args.get("published_after")
        published_before = args.get("published_before")
        try:
            after = date.fromisoformat(published_after) if published_after else None
            before = date.fromisoformat(published_before) if published_before else None
        except (TypeError, ValueError):
            return "Publication dates must use YYYY-MM-DD."
        if after and before and after >= before:
            return "published_after must be earlier than published_before."
        search_options = {"count": count}
        if after:
            search_options["published_after"] = after.isoformat()
        if before:
            search_options["published_before"] = before.isoformat()
        results = await search(query, domains, **search_options)
        results = [r for r in results if r.get("url")]
        results = results[:max(1, min(count, 10))]
        if not results:
            return abstain_msg
        # Tag with trust tier, then rank: preferred domains first, then authoritative→…→community.
        tagged = [(r, _tier_of(r["url"], source_tiers, news_tier)) for r in results]
        tagged.sort(key=lambda rt: (_prefers(rt[0]["url"], prefer), TIER_RANK.get(rt[1], 0)), reverse=True)

        blocks = []
        for r, tier in tagged:
            snippet = r.get("snippet", "")[:400]
            warning = archive_warning(r["url"], f"{r.get('title', '')}\n{snippet}")
            if warning:
                snippet = f"{warning}\n\n{snippet}"
            provenance = {"evidence_grade": "discovery"}
            if tier == "authoritative" and not warning:
                provenance = {
                    "evidence_grade": "authoritative_excerpt",
                    "source_tier": "authoritative",
                }
            elif tier == "unverified":
                provenance["source_tier"] = "unverified"
            cite = ctx.citations.register(
                r["url"],
                snippet=snippet,
                title=r.get("title", ""),
                kind="WEB",
                provenance=provenance,
            )
            label = _TIER_LABELS.get(tier, tier)
            blocks.append(
                f"[{cite}] ({label}) {r.get('title','')} ({r['url']})\n{snippet}"
            )
        guidance = (
            "\n\nYou may cite only claims directly supported by an official excerpt. "
            "For details beyond an excerpt, call web_fetch with its URL and a focused query. "
            "Editorial, news, community, and archived results remain discovery only."
            if any(tier == "authoritative" for _result, tier in tagged)
            else ""
        )
        return "\n\n".join(blocks) + guidance

    return _handler


def web_search_tools(
    allowlist: list[str],
    source_tiers: Optional[dict[str, tuple[str, str]]] = None,
    news_tier: Optional[list[str]] = None,
    search_fn: Optional[SearchFn] = None,
) -> list[Tool]:
    source_tiers = source_tiers or {}
    news_tier = news_tier or []
    search = search_fn or tavily_search
    search_domains = sorted(set(allowlist) | {d.lower() for d in news_tier})

    web_search = _make_handler(
        search, search_domains, source_tiers, news_tier,
        abstain_msg=(
            "No results from the live web for that query. Tell the user you couldn't verify it rather than "
            "guessing."
        ),
    )

    prefer_param = {
        "prefer": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "One domain to rank ahead of other returned results",
            },
            "description": (
                "Optional domains to rank first after search. This ranks only the results "
                "returned by the provider; it does not restrict domains or guarantee that "
                "a preferred domain is retrieved."
            ),
        },
    }
    publication_params = {
        "published_after": {
            "type": "string",
            "format": "date",
            "description": (
                "Optional exclusive lower bound for publication or last-update date in "
                "YYYY-MM-DD. Use this for publication freshness, not the date of an event "
                "described by the page."
            ),
        },
        "published_before": {
            "type": "string",
            "format": "date",
            "description": (
                "Optional exclusive upper bound for publication or last-update date in YYYY-MM-DD. "
                "Combine with published_after for a bounded publication range."
            ),
        },
    }
    count_param = {
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Maximum number of results to return, from 1 to 10.",
        },
    }
    return [
        Tool(
            name="web_search",
            description=(
                "Search the live web for current facts, current events, long-tail information, or an ambiguous "
                "reference that a structured NYC tool does not cover. Use a short noun-phrase query, "
                "optionally with NYC or a date, rather than the resident's whole sentence. Results "
                "rank known sources by trust but retain unlisted sources as leads. Authoritative "
                "excerpts may support only the claims stated in the excerpt; use `web_fetch` when "
                "the needed detail is on the result page. This tool is always available. For the "
                "same missing fact, make one focused search, then say you could not confirm it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    **prefer_param,
                    **publication_params,
                    **count_param,
                },
                "required": ["query"],
            },
            open_world=True,
            handler=web_search,
        ),
    ]
