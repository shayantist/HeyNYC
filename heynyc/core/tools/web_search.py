"""Scoped web search — the long-tail / fresh-info fallback.

Restricted to an allowlist of trusted NYC domains so the agent can't wander onto
random sources. Results are double-checked against the allowlist (defense in depth)
and registered as WEB citations. The search backend is injectable for tests; the
default uses Tavily (which supports include_domains) and degrades gracefully to
"unavailable" when no key is set, so the agent falls back to the index or abstains.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from .. import config
from ..registry import TIER_RANK
from .base import Tool, ToolContext

# (query, allowed_domains) -> list of {"title","url","snippet"}
SearchFn = Callable[[str, list[str]], Awaitable[list[dict]]]


def _domain_allowed(url: str, allowlist: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowlist)


async def tavily_search(query: str, allowed_domains: list[str]) -> list[dict]:
    """Default backend. Returns [] when no API key (caller treats as unavailable)."""
    if not config.TAVILY_API_KEY:
        return []
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "include_domains": allowed_domains,
                "max_results": 5,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in results]


_BASE_GOV = {"nyc.gov", "cityofnewyork.us", "mta.info"}


def _tier_of(url: str, source_tiers: dict[str, tuple[str, str]]) -> str:
    """Best tier for a URL's host: an explicit source_tiers match (highest wins),
    else a default — gov domains are authoritative, everything allowlisted is editorial."""
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
    return "editorial"


def _prefers(url: str, prefer: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in prefer)


def web_search_tools(
    allowlist: list[str],
    source_tiers: Optional[dict[str, tuple[str, str]]] = None,
    search_fn: Optional[SearchFn] = None,
) -> list[Tool]:
    search = search_fn or tavily_search
    source_tiers = source_tiers or {}

    async def _handler(args: dict, ctx: ToolContext) -> str:
        prefer = args.get("prefer") or []
        results = await search(args["query"], allowlist)
        # Defense in depth: drop anything outside the allowlist even if the provider slipped it in.
        results = [r for r in results if r.get("url") and _domain_allowed(r["url"], allowlist)]
        if not results:
            return (
                "No results from trusted NYC sources for that query. "
                "Tell the user you couldn't find it on official sources rather than guessing."
            )
        # Tag with trust tier, then rank: preferred domains first, then authoritative→community.
        tagged = [(r, _tier_of(r["url"], source_tiers)) for r in results]
        tagged.sort(key=lambda rt: (_prefers(rt[0]["url"], prefer), TIER_RANK.get(rt[1], 0)), reverse=True)

        blocks = []
        for r, tier in tagged:
            cite = ctx.citations.register(
                r["url"], snippet=r.get("snippet", "")[:200], title=r.get("title", ""), kind="WEB"
            )
            label = "⚠️ community-posted — confirm before you go" if tier == "community" else tier
            blocks.append(
                f"[{cite}] ({label}) {r.get('title','')} ({r['url']})\n{r.get('snippet','')[:400]}"
            )
        return "\n\n".join(blocks)

    return [
        Tool(
            name="web_search",
            description=(
                "Search trusted NYC web sources (nyc.gov, nyctourism.com, official event sites, etc.) "
                "for fresh or long-tail info not in the index — e.g. a specific event this weekend. "
                "Restricted to an allowlist and ranked by source trust; results are tagged "
                "authoritative/editorial/community. Treat community-tagged (⚠️) results as unconfirmed "
                "and tell the user to verify. Pass `prefer` to boost the active topic's official "
                "domains. Cite every result; if nothing comes back, abstain."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "prefer": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional domains to rank first (e.g. a topic's official sites).",
                    },
                },
                "required": ["query"],
            },
            open_world=True,  # hits the open web (allowlisted)
            handler=_handler,
        )
    ]
