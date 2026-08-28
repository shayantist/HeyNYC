"""Web search for fresh and long-tail information.

Known domains are trust metadata, not a retrieval filter. The search backend is
injectable for tests; production uses Tavily, then Brave, then DuckDuckGo, and
marks unknown sources as unverified leads rather than silently discarding them.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import date
from typing import Annotated, Awaitable, Callable, NotRequired, Optional, TypedDict
from urllib.parse import urlparse

import httpx
from pydantic import Field, model_validator

from .. import config
from ..registry import TIER_RANK
from .base import Tool, ToolContext, ToolInput

# (query, allowed_domains, published_after=None, published_before=None, count=20,
#  include_domains=None)
# -> result dictionaries

class RankedWebResult(TypedDict):
    """Normalized provider result retained before answer evidence is packed."""

    title: str
    url: str
    snippet: str
    search_provider: NotRequired[str]
    score: NotRequired[float]
    published_date: NotRequired[str]
    page_age: NotRequired[str]
    publisher: NotRequired[str]
    raw_content: NotRequired[str]
    content_scope: NotRequired[str]
    source_tier: NotRequired[str]
    child_links: NotRequired[list[dict]]
    acquisition: NotRequired[dict]
    hydration_error: NotRequired[str]
    _rrf_score: NotRequired[float]


SearchFn = Callable[..., Awaitable[list[RankedWebResult]]]
logger = logging.getLogger("heynyc.web_search")
_tavily_plan_exhausted = False
_PROVIDER_RESULT_COUNT = 20  # Tavily and Brave Web Search maximum per request.

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


class SearchUnavailable(RuntimeError):
    """Every configured search provider failed before returning a valid response."""


class WebSearchInput(ToolInput):
    queries: list[Annotated[str, Field(description="Focused search")]] = Field(
        min_length=1,
        description="Independent focused searches",
    )
    domains: list[str] | None = Field(default=None, description="Restrict results to domains")
    published_after: date | None = Field(default=None, description="Publication date lower bound")
    published_before: date | None = Field(default=None, description="Publication date upper bound")

    @model_validator(mode="after")
    def valid_publication_window(self):
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after >= self.published_before
        ):
            raise ValueError("published_after must be earlier than published_before")
        return self


def _query_list(args: dict) -> list[str]:
    values = args.get("queries")
    if values is None and args.get("query") is not None:
        values = [args["query"]]
    return list(dict.fromkeys(str(value).strip() for value in values or [] if str(value).strip()))


def _fuse_query_results(groups: list[list[dict]]) -> list[dict]:
    fused: dict[str, dict] = {}
    for group in groups:
        for rank, result in enumerate(group):
            url = str(result.get("url") or "")
            if not url:
                continue
            score = 1 / (60 + rank)
            if url not in fused:
                fused[url] = {**result, "_rrf_score": score}
                continue
            fused[url]["_rrf_score"] = fused[url].get("_rrf_score", 0.0) + score
            snippets = [
                str(fused[url].get("snippet") or "").strip(),
                str(result.get("snippet") or "").strip(),
            ]
            fused[url]["snippet"] = "\n\n".join(dict.fromkeys(s for s in snippets if s))
    return sorted(fused.values(), key=lambda result: result["_rrf_score"], reverse=True)


def _normalized_domains(domains: list[str]) -> list[str]:
    return sorted({
        domain.strip().lower().lstrip(".")
        for domain in domains
        if domain.strip()
    })


def _domain_allowed(url: str, allowlist: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == domain or host.endswith("." + domain)
        for domain in _normalized_domains(allowlist)
    )


def archive_warning(url: str, text: str = "") -> str:
    path = (urlparse(url).path or "").lower()
    haystack = text.lower()
    if "/archive/" in path or "save-policy-news-archive" in path:
        return _ARCHIVE_WARNING
    return _ARCHIVE_WARNING if any(marker in haystack for marker in _ARCHIVE_MARKERS) else ""


async def _tavily(query: str, allowed_domains: list[str], **extra) -> list[dict]:
    """Shared Tavily call."""
    if not config.TAVILY_API_KEY:
        raise SearchUnavailable("Tavily is not configured")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "max_results": _PROVIDER_RESULT_COUNT,
                    "search_depth": "basic",
                    **extra,
                },
                headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", []) if isinstance(payload, dict) else []
    except (httpx.TimeoutException, httpx.TransportError, TypeError, ValueError) as exc:
        logger.warning("Tavily search transport unavailable: %s", type(exc).__name__)
        raise SearchUnavailable(f"Tavily failed with {type(exc).__name__}") from exc
    if not isinstance(results, list):
        raise SearchUnavailable("Tavily returned an invalid response")
    normalized = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = str(result.get("content") or "").strip()
        row = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": content,
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
        raw_content = result.get("raw_content")
        if isinstance(raw_content, str) and raw_content.strip():
            row["raw_content"] = raw_content.strip()
            row["content_scope"] = "provider_extract"
        normalized.append(row)
    return normalized


async def _brave(
    query: str,
    allowed_domains: list[str],
    *,
    count: int = _PROVIDER_RESULT_COUNT,
    include_domains: Optional[list[str]] = None,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    **_options,
) -> list[dict]:
    """Brave ranked web results for discovery before selective page reading."""
    if not config.BRAVE_SEARCH_API_KEY:
        raise SearchUnavailable("Brave is not configured")
    if bool(published_after) != bool(published_before):
        raise SearchUnavailable("Brave requires both publication bounds")
    request: dict[str, object] = {
        "q": query,
        "count": count,
        "extra_snippets": True,
    }
    if include_domains:
        request["goggles"] = "\n".join(
            f"$boost,site={domain}" for domain in include_domains
        )
    if published_after and published_before:
        request["freshness"] = f"{published_after}to{published_before}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=request,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": config.BRAVE_SEARCH_API_KEY,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except (
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.TransportError,
        TypeError,
        ValueError,
    ) as exc:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        logger.warning(
            "Brave search fallback unavailable: %s%s",
            type(exc).__name__,
            f" status={status}" if status is not None else "",
        )
        raise SearchUnavailable(f"Brave failed with {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise SearchUnavailable("Brave returned an invalid response")
    web = payload.get("web") or {}
    if not isinstance(web, dict):
        raise SearchUnavailable("Brave returned an invalid web response")
    results = web.get("results") or []
    if not isinstance(results, list):
        raise SearchUnavailable("Brave returned invalid results")
    normalized = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url", "")
        snippets = [str(result.get("description") or "").strip()]
        extra_snippets = result.get("extra_snippets") or []
        if isinstance(extra_snippets, list):
            snippets.extend(str(item).strip() for item in extra_snippets)
        snippet = "\n\n".join(dict.fromkeys(item for item in snippets if item))
        if not url:
            continue
        row = {
            "title": result.get("title", ""),
            "url": url,
            "snippet": snippet,
            "search_provider": "Brave Web Search API",
        }
        if result.get("page_age"):
            row["page_age"] = str(result["page_age"])
        profile = result.get("profile") or {}
        if isinstance(profile, dict) and (profile.get("long_name") or profile.get("name")):
            row["publisher"] = str(profile.get("long_name") or profile["name"])
        normalized.append(row)
    return normalized


async def _duckduckgo(
    query: str,
    allowed_domains: list[str],
    count: int = _PROVIDER_RESULT_COUNT,
    topic: Optional[str] = None,
) -> list[dict]:
    """Local search fallback for a Tavily plan-limit response."""
    try:
        from ddgs import DDGS
    except Exception as exc:
        raise SearchUnavailable(f"DuckDuckGo failed with {type(exc).__name__}") from exc

    if topic == "news":
        try:
            results = await asyncio.to_thread(
                DDGS().news, query, max_results=count, backend="duckduckgo",
            )
        except Exception as exc:
            logger.warning(
                "DDGS DuckDuckGo news unavailable; trying text search: %s",
                type(exc).__name__,
            )
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
                for result in results or []
                if isinstance(result, dict) and result.get("url")
            ]
    try:
        results = await asyncio.to_thread(
            DDGS().text, query, max_results=count, backend="duckduckgo",
        )
    except Exception as exc:
        logger.warning("DDGS DuckDuckGo text search unavailable: %s", type(exc).__name__)
        raise SearchUnavailable(f"DuckDuckGo failed with {type(exc).__name__}") from exc
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", ""),
        }
        for result in results or []
        if isinstance(result, dict) and result.get("href")
    ]


async def _search_with_fallback(
    query: str,
    allowed_domains: list[str],
    *,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    include_domains: Optional[list[str]] = None,
    count: int = _PROVIDER_RESULT_COUNT,
    topic: Optional[str] = None,
    **tavily_options,
) -> list[dict]:
    global _tavily_plan_exhausted

    async def secondary(failures: list[str] | None = None) -> list[dict]:
        failures = list(failures or [])
        try:
            results = await _brave(
                query, allowed_domains, published_after=published_after,
                published_before=published_before, include_domains=include_domains,
                count=count, topic=topic, **tavily_options,
            )
        except SearchUnavailable as exc:
            failures.append(str(exc))
            results = []
        provider = "Brave Web Search API"
        if not results:
            failures.append("Brave returned no results")
            try:
                results = await _duckduckgo(
                    query, allowed_domains, count=count, topic=topic,
                )
            except SearchUnavailable as exc:
                failures.append(str(exc))
                raise SearchUnavailable("; ".join(failures)) from exc
            provider = "DuckDuckGo"
        for result in results:
            result.pop("_rrf_score", None)
        for result in results:
            result.setdefault("search_provider", provider)
            if failures:
                result["degraded_providers"] = failures
            if topic in {"news", "finance"} and result["search_provider"] in {
                "Brave Web Search API", "DuckDuckGo",
            }:
                result["degraded_from_topic"] = topic
            if published_after or published_before:
                result["degraded_publication_bounds"] = {
                    "published_after": published_after,
                    "published_before": published_before,
                }
        return results

    if _tavily_plan_exhausted:
        return await secondary(["Tavily plan exhausted"])
    try:
        options = dict(tavily_options)
        if options.pop("include_page_evidence", False):
            options["search_depth"] = "advanced"
            options["include_raw_content"] = True
        if topic:
            options["topic"] = topic
        results = await _tavily(
            query,
            allowed_domains,
            max_results=count,
            **({"include_domains": include_domains} if include_domains else {}),
            **options,
        )
    except (
        SearchUnavailable,
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 432:
            _tavily_plan_exhausted = True
            logger.warning("Tavily plan exhausted; using secondary web search")
        else:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning(
                "Tavily search unavailable; using secondary web search: %s%s",
                type(exc).__name__,
                f" status={status}" if status is not None else "",
            )
        return await secondary([str(exc)])
    return results or await secondary(["Tavily returned no results"])


async def tavily_search(
    query: str,
    allowed_domains: list[str],
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    count: int = _PROVIDER_RESULT_COUNT,
    include_domains: Optional[list[str]] = None,
    topic: Optional[str] = None,
    **search_options,
) -> list[dict]:
    """Shared Tavily, Brave, then DuckDuckGo search with optional date bounds."""
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
        **search_options,
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
    "unverified": "search excerpt, cite only what it states",
}


def _text_tokens(text: str, model: str | None = None) -> int:
    import litellm

    return int(litellm.token_counter(model=model or config.HEYNYC_MODEL, text=text))


async def _hydrate_search_result(
    result: dict,
    query: str,
    ctx: ToolContext,
) -> dict:
    """Read one fused search result without starting a browser or another search."""
    if result.get("acquisition"):
        return result
    from .web_fetch import _fetch_page, _line_addressable

    try:
        fetched = await _fetch_page(result["url"], ctx.http)
    except Exception as exc:
        return {**result, "hydration_error": type(exc).__name__}
    return {
        **result,
        "raw_content": _line_addressable(fetched.text) or str(result.get("snippet") or ""),
        "child_links": fetched.child_links,
        "acquisition": fetched.acquisition.model_dump(mode="json"),
    }


def _hydration_candidates(
    results: list[dict],
    needs_hydration: Callable[[dict], bool] | None = None,
) -> list[int]:
    """Choose only results that still need a page read."""
    selected: list[int] = []
    for index, result in enumerate(results):
        if needs_hydration is not None:
            if needs_hydration(result):
                selected.append(index)
            continue
        if result.get("acquisition") or result.get("content_scope") in {
            "provider_context", "provider_extract",
        }:
            continue
        selected.append(index)
    return selected


async def batch_hydrate_results(
    results: list[dict],
    query: str,
    ctx: ToolContext,
) -> list[dict]:
    """Hydrate fused results concurrently, relying on each fetch's network timeout."""
    hydrated_results = list(results)
    selected = _hydration_candidates(hydrated_results)
    if not selected:
        return hydrated_results

    async def hydrate(index: int) -> tuple[int, dict]:
        return index, await _hydrate_search_result(hydrated_results[index], query, ctx)

    for index, result in await asyncio.gather(*(hydrate(index) for index in selected)):
        hydrated_results[index] = result
    return hydrated_results


async def hydrate_ranked_results(
    results: list[dict],
    query: str,
    ctx: ToolContext,
    *,
    policy: str,
    sufficient: Callable[[list[dict]], bool],
    needs_hydration: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Hydrate ranked pages until the caller's deterministic sufficiency test passes."""
    if policy not in {"fast", "deep"}:
        raise ValueError("policy must be fast or deep")
    hydrated = list(results)
    selected = _hydration_candidates(hydrated, needs_hydration)
    started = time.perf_counter()
    completed = 0
    stopping_reason = "exhausted"
    async def run() -> None:
        nonlocal completed, stopping_reason
        if policy == "fast":
            for index in selected:
                hydrated[index] = await _hydrate_search_result(hydrated[index], query, ctx)
                completed += 1
                if sufficient(hydrated):
                    stopping_reason = "sufficient"
                    break
            return
        tasks = {
            index: asyncio.create_task(_hydrate_search_result(hydrated[index], query, ctx))
            for index in selected
        }
        try:
            for index in selected:
                hydrated[index] = await tasks[index]
                completed += 1
                if sufficient(hydrated):
                    stopping_reason = "sufficient"
                    break
        finally:
            pending = [task for task in tasks.values() if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    try:
        if policy == "deep":
            async with asyncio.timeout(60):
                await run()
        else:
            await run()
    except TimeoutError:
        stopping_reason = "deadline"
    ctx.tool_runs.append({
        "operation": "web_hydration",
        "policy": policy,
        "candidates": len(results),
        "eligible": len(selected),
        "completed": completed,
        "stopping_reason": stopping_reason,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    })
    return hydrated


async def retrieve_ranked_results(
    args: dict,
    ctx: ToolContext,
    *,
    search: SearchFn | None = None,
    domains: list[str] | None = None,
    source_tiers: dict[str, tuple[str, str]] | None = None,
    news_tier: list[str] | None = None,
) -> list[dict]:
    """Return normalized ranked web results before answer-facing rendering."""
    search = search or tavily_search
    news_tier = ctx.registry.news_tier() if news_tier is None else news_tier
    source_tiers = ctx.registry.source_tiers() if source_tiers is None else source_tiers
    if domains is None:
        domains = sorted(set(ctx.registry.allowlist()) | {d.lower() for d in news_tier})
    prefer = args.get("prefer") or []
    restrict_domains = _normalized_domains(args.get("restrict_domains") or [])
    queries = _query_list(args)
    if not queries:
        raise ValueError("At least one search query is required.")
    query = queries[0]
    published_after = args.get("published_after")
    published_before = args.get("published_before")
    topic = args.get("topic")
    include_page_evidence = args.get("include_page_evidence") is True
    relevance_first = ctx.event_turn == "discovery" and not ctx.current_turn_high_stakes
    if topic not in {None, "general", "news", "finance"}:
        raise ValueError("topic must be general, news, or finance.")
    try:
        after = date.fromisoformat(published_after) if published_after else None
        before = date.fromisoformat(published_before) if published_before else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Publication dates must use YYYY-MM-DD.") from exc
    if after and before and after >= before:
        raise ValueError("published_after must be earlier than published_before.")
    search_options = {}
    if after:
        search_options["published_after"] = after.isoformat()
    if before:
        search_options["published_before"] = before.isoformat()
    if topic:
        search_options["topic"] = topic
    if include_page_evidence:
        search_options["include_page_evidence"] = True
    search_started = time.perf_counter()
    results = await search(
        query,
        domains,
        **({"include_domains": restrict_domains} if restrict_domains else {}),
        **search_options,
    )
    from .web_fetch import _line_addressable, _url_safe_shape, _validate_public_url

    results = [
        result for result in results
        if result.get("url") and _url_safe_shape(str(result["url"]))
    ]
    if restrict_domains:
        results = [
            result for result in results
            if _domain_allowed(str(result["url"]), restrict_domains)
        ]

    async def public(result: dict) -> dict | None:
        if not result.get("search_provider"):
            return result
        try:
            await _validate_public_url(str(result["url"]))
        except ValueError:
            return None
        return result

    results = [
        result
        for result in await asyncio.gather(*(public(result) for result in results))
        if result is not None
    ]
    for result in results:
        if result.get("content_scope") != "provider_extract":
            continue
        result["raw_content"] = _line_addressable(
            str(result.get("raw_content") or "")
        ) or str(result.get("snippet") or "")

    def page_key(result: dict) -> tuple[str, str]:
        parsed = urlparse(result["url"])
        return (
            (parsed.hostname or "").casefold(),
            parsed.path.rstrip("/").casefold(),
        )

    canonical_pages = {
        page_key(result)
        for result in results
        if urlparse(result["url"]).query == ""
    }
    merged: dict[tuple[str, ...], dict] = {}
    for result in results:
        parsed = urlparse(result["url"])
        base_key = page_key(result)
        key = base_key if base_key in canonical_pages else (*base_key, parsed.query)
        existing = merged.get(key)
        if existing is None or parsed.query == "":
            merged[key] = dict(result)
    results = list(merged.values())
    if include_page_evidence:
        results = await batch_hydrate_results(results, query, ctx)
    for result in results:
        result["source_tier"] = _tier_of(result["url"], source_tiers, news_tier)
    if not relevance_first:
        results.sort(
            key=lambda result: (
                TIER_RANK.get(result["source_tier"], 0),
                _prefers(result["url"], prefer),
                result.get("_rrf_score", -1.0),
                result.get("score", -1.0),
            ),
            reverse=True,
        )
    ctx.tool_runs.append({
        "operation": "web_retrieval",
        "provider_latency_ms": round((time.perf_counter() - search_started) * 1000),
        "retrieved_candidates": len(results),
        "normalized_candidates": len(results),
    })
    return results


def search_result_evidence(result: dict, ctx: ToolContext) -> tuple[str, dict, str]:
    """Build one source excerpt and provenance record from a normalized result."""
    tier = str(result.get("source_tier") or "unverified")
    raw_content = str(result.get("raw_content") or "").strip()
    snippet = raw_content or result.get("snippet", "")
    content_missing = not str(snippet).strip()
    if content_missing:
        snippet = "No page content was retrieved."
    warning = archive_warning(result["url"], f"{result.get('title', '')}\n{snippet}")
    if warning:
        snippet = f"{warning}\n\n{snippet}"
    degraded = bool(
        result.get("degraded_publication_bounds") or result.get("degraded_from_topic")
    )
    display_tier = "unverified" if degraded or content_missing else tier
    provenance = {"evidence_grade": "discovery"}
    if degraded or content_missing:
        provenance["source_tier"] = "unverified"
    if raw_content and result.get("acquisition") and not warning and not degraded:
        provenance = {"evidence_grade": "fetched", "source_tier": tier}
    elif tier == "authoritative" and not warning and not degraded and not content_missing:
        provenance = {
            "evidence_grade": "authoritative_excerpt",
            "source_tier": "authoritative",
        }
    elif tier in {"editorial", "news"} and not warning and not degraded and not content_missing:
        provenance = {"evidence_grade": "search_excerpt", "source_tier": tier}
    elif (
        tier == "community"
        and not warning
        and ctx.allow_unverified_search_excerpts
        and not ctx.current_turn_high_stakes
        and not degraded
        and not content_missing
    ):
        provenance = {"evidence_grade": "search_excerpt", "source_tier": tier}
    elif (
        tier == "unverified"
        and not warning
        and ctx.allow_unverified_search_excerpts
        and not ctx.current_turn_high_stakes
        and not degraded
        and not content_missing
    ):
        provenance = {"evidence_grade": "search_excerpt", "source_tier": "unverified"}
    elif tier == "unverified":
        provenance["source_tier"] = "unverified"
    search_metadata = {
        **({"provider": result["search_provider"]} if result.get("search_provider") else {}),
        **({"score": result["score"]} if "score" in result else {}),
        **(
            {"published_date": result["published_date"]}
            if result.get("published_date") else {}
        ),
        **({"page_age": result["page_age"]} if result.get("page_age") else {}),
        **({"publisher": result["publisher"]} if result.get("publisher") else {}),
        **(
            {"degraded_providers": result["degraded_providers"]}
            if result.get("degraded_providers") else {}
        ),
        **(
            {"degraded_from_topic": result["degraded_from_topic"]}
            if result.get("degraded_from_topic") else {}
        ),
        **(
            {"degraded_publication_bounds": result["degraded_publication_bounds"]}
            if result.get("degraded_publication_bounds") else {}
        ),
    }
    if search_metadata:
        provenance["search"] = search_metadata
        metadata = []
        if result.get("published_date"):
            metadata.append(f"Published: {result['published_date']}")
        if result.get("page_age"):
            metadata.append(f"Page date: {result['page_age']}")
        if result.get("publisher"):
            metadata.append(f"Publisher: {result['publisher']}")
        snippet = "\n".join([*metadata, snippet])
    if result.get("child_links"):
        provenance["child_links"] = result["child_links"]
    if result.get("acquisition"):
        provenance["acquisition"] = result["acquisition"]
    if result.get("hydration_error"):
        provenance["acquisition_failure"] = result["hydration_error"]
        snippet = (
            f"{snippet}\nPage read failed ({result['hydration_error']}); "
            "only the search excerpt was retrieved."
        ).strip()
    return snippet, provenance, _TIER_LABELS.get(display_tier, display_tier)


def _make_handler(
    search: SearchFn,
    domains: list[str],
    source_tiers: dict[str, tuple[str, str]],
    news_tier: list[str],
    *,
    abstain_msg: str,
) -> Callable:
    """Build a search handler over `domains`, tagging + ranking results by trust tier."""
    async def _handler(args: WebSearchInput, ctx: ToolContext) -> str:
        queries = _query_list(args)
        requested_domains = args.get("domains")
        try:
            outcomes = await asyncio.gather(*(
                retrieve_ranked_results(
                    {
                        **{
                            key: value
                            for key, value in args.items()
                            if key not in {"queries", "domains"}
                        },
                        "query": query,
                        **(
                            {"restrict_domains": requested_domains}
                            if requested_domains else {}
                        ),
                    },
                    ctx,
                    search=search,
                    domains=domains,
                    source_tiers=source_tiers,
                    news_tier=news_tier,
                )
                for query in queries
            ), return_exceptions=True)
        except ValueError as exc:
            return str(exc)
        groups = [outcome if isinstance(outcome, list) else [] for outcome in outcomes]
        validation_error = next(
            (outcome for outcome in outcomes if isinstance(outcome, ValueError)),
            None,
        )
        if validation_error is not None:
            return str(validation_error)
        failures = [
            (query, outcome)
            for query, outcome in zip(queries, outcomes)
            if isinstance(outcome, SearchUnavailable)
        ]
        unexpected = next(
            (
                outcome
                for outcome in outcomes
                if isinstance(outcome, Exception)
                and not isinstance(outcome, SearchUnavailable)
            ),
            None,
        )
        if unexpected is not None:
            raise unexpected
        if failures and not any(groups):
            from .base import ToolFailure
            return ToolFailure(
                status="unavailable",
                reason="Web search providers are unavailable: " + "; ".join(
                    str(error) for _query, error in failures
                ),
                retryable=True,
            )
        if not any(groups):
            return abstain_msg
        tagged = [
            (query, result, result["source_tier"])
            for query, group in zip(queries, groups)
            for result in group
        ]
        degraded_topics = sorted({
            r["degraded_from_topic"]
            for _query, r, _tier in tagged
            if r.get("degraded_from_topic")
        })
        degraded_bounds = any(
            r.get("degraded_publication_bounds") for _query, r, _tier in tagged
        )
        search_context = []
        search_context.extend(
            f"QUERY FAILED: {query}\n{error}"
            for query, error in failures
        )
        degraded_providers = sorted({
            provider
            for _query, result, _tier in tagged
            for provider in result.get("degraded_providers") or []
        })
        if degraded_providers:
            search_context.append(
                "Search used a fallback after: " + "; ".join(degraded_providers)
            )
        if degraded_topics:
            search_context.append(
                f"Requested {', '.join(degraded_topics)} search was unavailable; this is a "
                "general fallback without provider publication-date guarantees."
            )
        if degraded_bounds:
            search_context.append(
                "The final fallback could not enforce the requested publication-date bounds; "
                "treat these as possible leads and verify their dates from the linked pages."
            )

        prepared = []
        for query, r, tier in tagged:
            snippet, provenance, label = search_result_evidence(r, ctx)
            prepared.append((query, r, snippet, provenance, label))

        available = ctx.evidence_token_budget
        model = ctx.evidence_model or config.HEYNYC_MODEL
        blocks = []
        current_query = None
        for query, r, snippet, provenance, label in prepared:
            header = f"QUERY: {query}\n" if query != current_query else ""
            block = f"{header}[S0] ({label}) {r.get('title','')} ({r['url']})\n{snippet}"
            needed = _text_tokens(block, model)
            if available is not None and needed > available:
                continue
            cite = ctx.citations.register(
                r["url"],
                snippet=snippet,
                title=r.get("title", ""),
                kind="WEB",
                provenance=provenance,
            )
            blocks.append(
                f"{header}[{cite}] ({label}) {r.get('title','')} ({r['url']})\n{snippet}"
            )
            current_query = query
            if available is not None:
                available -= needed
        if ctx.evidence_token_budget is not None:
            ctx.evidence_tokens_used += ctx.evidence_token_budget - available
            ctx.evidence_token_budget = available
        omitted = len(prepared) - len(blocks)
        if omitted:
            if blocks:
                search_context.append(
                    f"Search returned {len(prepared)} ranked pages; evidence from {len(blocks)} fit "
                    "this turn's evidence budget. Answer from the included evidence; a narrower "
                    "resident follow-up can inspect other pages."
                )
            else:
                result = prepared[0][1]
                cite = ctx.citations.register(
                    result["url"],
                    snippet="No evidence fit the remaining model context.",
                    title=result.get("title", ""),
                    kind="WEB",
                    provenance={"evidence_grade": "discovery"},
                )
                search_context.append(
                    f"No evidence fit the remaining model context. Highest-ranked source retained: "
                    f"[{cite}] {result.get('title', '')} ({result['url']})."
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

    async def ranked_results(args: WebSearchInput, ctx: ToolContext) -> list[dict]:
        groups = await asyncio.gather(*(
            retrieve_ranked_results(
                {
                    **{
                        (
                            "restrict_domains" if key == "domains" else key
                        ): value
                        for key, value in args.items()
                        if key != "queries"
                    },
                    "query": query,
                },
                ctx,
                search=search,
                domains=search_domains,
                source_tiers=source_tiers,
                news_tier=news_tier,
            )
            for query in _query_list(args)
        ))
        return _fuse_query_results(groups)
    return [
        Tool(
            name="web_search",
            description=(
                "Search the live web for current facts, events, long-tail information, or an "
                "ambiguous reference. Use a short noun-phrase query. Results are ranked source "
                "excerpts with provenance metadata. Authoritative excerpts and curated "
                "editorial/news excerpts support only the claims they state. Use `web_fetch` "
                "when a promising excerpt does not contain the needed detail."
            ),
            input_type=WebSearchInput,
            open_world=True,
            handler=web_search,
            result_handler=ranked_results,
        ),
    ]
