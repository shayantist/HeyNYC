"""Web search for fresh and long-tail information.

Known domains are trust metadata, not a retrieval filter. The search backend is
injectable for tests; production uses Tavily Basic and marks unknown sources as
unverified leads rather than silently discarding them.
"""
from __future__ import annotations

import asyncio
import math
from datetime import date
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from .. import config
from ..registry import TIER_RANK
from .base import Tool, ToolContext

# (query, allowed_domains, published_after=None, published_before=None, count=5,
#  include_domains=None)
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
    normalized = []
    for result in results:
        row = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", ""),
            "search_provider": "Tavily Search API",
        }
        score = result.get("score")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
        ):
            row["score"] = float(score)
        published_date = result.get("published_date")
        if isinstance(published_date, str) and published_date.strip():
            row["published_date"] = published_date.strip()
        normalized.append(row)
    return normalized


async def _duckduckgo(
    query: str,
    allowed_domains: list[str],
    count: int = 5,
    topic: Optional[str] = None,
) -> list[dict]:
    """Local search fallback for a Tavily plan-limit response."""
    from ddgs import DDGS

    if topic == "news":
        try:
            results = await asyncio.to_thread(DDGS().news, query, max_results=count)
        except Exception:
            pass
        else:
            return [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "search_provider": "DuckDuckGo News",
                    **(
                        {"published_date": result["date"]}
                        if result.get("date")
                        else {}
                    ),
                    **({"publisher": result["source"]} if result.get("source") else {}),
                }
                for result in results
                if result.get("url")
            ][:count]
    try:
        results = await asyncio.to_thread(DDGS().text, query, max_results=count)
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
    include_domains: Optional[list[str]] = None,
    count: int = 5,
    topic: Optional[str] = None,
    **tavily_options,
) -> list[dict]:
    try:
        options = dict(tavily_options)
        if include_domains:
            options["include_domains"] = include_domains
        if topic:
            options["topic"] = topic
        return await _tavily(query, allowed_domains, max_results=count, **options)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 432:
            raise
        if published_after or published_before:
            return []
        fallback_query = query
        if include_domains:
            sites = " OR ".join(f"site:{domain}" for domain in include_domains)
            fallback_query = f"{query} ({sites})"
        results = await _duckduckgo(
            fallback_query,
            allowed_domains,
            count=count,
            topic=topic,
        )
        for result in results:
            result.setdefault("search_provider", "DuckDuckGo")
            if topic in {"news", "finance"} and result["search_provider"] == "DuckDuckGo":
                result["degraded_from_topic"] = topic
        return results


async def tavily_search(
    query: str,
    allowed_domains: list[str],
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    count: int = 5,
    include_domains: Optional[list[str]] = None,
    topic: Optional[str] = None,
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
        include_domains=include_domains,
        count=count,
        topic=topic,
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
    "editorial": "editorial source, cite only what the excerpt states",
    "news": "📰 news source, cite only what the excerpt states",
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
        topic = args.get("topic")
        relevance_first = ctx.event_turn == "discovery" and not ctx.current_turn_high_stakes
        if topic not in {None, "general", "news", "finance"}:
            return "topic must be general, news, or finance."
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
        if topic:
            search_options["topic"] = topic
        results = await search(query, domains, **search_options)
        results = [r for r in results if r.get("url")]
        if prefer and not any(_prefers(r["url"], prefer) for r in results):
            preferred = await search(
                query,
                domains,
                include_domains=prefer,
                **search_options,
            )
            seen = set()
            results = [
                result
                for result in [*preferred, *results]
                if result.get("url") and not (result["url"] in seen or seen.add(result["url"]))
            ]
        results = results[:max(1, min(count, 10))]
        if not results:
            return abstain_msg
        # Trust leads high-stakes lookups. Low-stakes discovery may explicitly lead with the
        # provider's relevance score while retaining the same evidence labels.
        tagged = [(r, _tier_of(r["url"], source_tiers, news_tier)) for r in results]
        tagged.sort(
            key=lambda rt: (
                _prefers(rt[0]["url"], prefer),
                *(
                    (rt[0].get("score", -1.0), TIER_RANK.get(rt[1], 0))
                    if relevance_first
                    else (TIER_RANK.get(rt[1], 0), rt[0].get("score", -1.0))
                ),
            ),
            reverse=True,
        )

        providers = sorted({r["search_provider"] for r, _tier in tagged if r.get("search_provider")})
        degraded_topics = sorted({
            r["degraded_from_topic"] for r, _tier in tagged if r.get("degraded_from_topic")
        })
        search_context = []
        if providers:
            search_context.append(f"Search provider: {', '.join(providers)}")
        if degraded_topics:
            search_context.append(
                f"Requested {', '.join(degraded_topics)} search was unavailable; this is a "
                "general fallback without provider publication-date guarantees."
            )

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
            elif tier in {"editorial", "news"} and not warning:
                provenance = {
                    "evidence_grade": "search_excerpt",
                    "source_tier": tier,
                }
            elif (
                tier == "unverified"
                and not warning
                and ctx.allow_unverified_search_excerpts
            ):
                provenance = {
                    "evidence_grade": "search_excerpt",
                    "source_tier": "unverified",
                }
            elif tier == "unverified":
                provenance["source_tier"] = "unverified"
            search_metadata = {
                **(
                    {"provider": r["search_provider"]}
                    if r.get("search_provider")
                    else {}
                ),
                **({"score": r["score"]} if "score" in r else {}),
                **(
                    {"published_date": r["published_date"]}
                    if r.get("published_date")
                    else {}
                ),
                **({"publisher": r["publisher"]} if r.get("publisher") else {}),
                **(
                    {"degraded_from_topic": r["degraded_from_topic"]}
                    if r.get("degraded_from_topic")
                    else {}
                ),
            }
            if search_metadata:
                provenance["search"] = search_metadata
                metadata = []
                if r.get("published_date"):
                    metadata.append(f"Published: {r['published_date']}")
                if r.get("publisher"):
                    metadata.append(f"Publisher: {r['publisher']}")
                if "score" in r:
                    metadata.append(
                        f"Provider relevance score: {r['score']} (not truth confidence)"
                    )
                snippet = "\n".join([*metadata, snippet])
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
        return "\n\n".join([*search_context, *blocks])

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
        abstain_msg="No results from the live web for that query.",
    )

    prefer_param = {
        "prefer": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "One domain to rank ahead of other returned results",
            },
            "description": (
                "Optional domains to rank first. When none appear in the initial results, "
                "one targeted search tries those domains and merges its results; it does not discard "
                "unlisted sources."
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
    topic_param = {
        "topic": {
            "type": "string",
            "enum": ["general", "news", "finance"],
            "description": (
                "Optional Tavily search category. Use `news` for recent reporting, sports, "
                "politics, or major current events and to receive provider publication dates. "
                "Use `finance` for financial-market reporting. Omit for the broad `general` "
                "default, including current official pages."
            ),
        },
    }
    return [
        Tool(
            name="web_search",
            description=(
                "Search the live web for current facts, current events, long-tail information, or "
                "an ambiguous reference. This tool is always available. Use a short noun-phrase "
                "query, optionally with NYC or a date, rather than the resident's whole sentence. "
                "Results rank known sources by trust but retain unlisted sources. Authoritative "
                "excerpts and curated editorial/news excerpts support only the claims they state. "
                "For an explicitly low-stakes capability, an unverified excerpt may support only "
                "the exact claim it states and must keep its source warning. Use `web_fetch` when "
                "the result page contains the needed detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A short search query for one fact-finding objective, including the constraints "
                            "that change that fact. For multiple independent facts, call `web_search` in "
                            "parallel with one focused query per fact. "
                            "Use `prefer` instead of `site:` so other useful sources remain discoverable."
                        ),
                    },
                    **prefer_param,
                    **publication_params,
                    **topic_param,
                    **count_param,
                },
                "required": ["query"],
            },
            open_world=True,
            handler=web_search,
        ),
    ]
