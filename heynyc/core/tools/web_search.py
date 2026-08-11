"""Web search for fresh and long-tail information.

Known domains are trust metadata, not a retrieval filter. The search backend is
injectable for tests; production uses Tavily Basic and marks unknown sources as
unverified leads rather than silently discarding them.
"""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from .. import config
from ..registry import TIER_RANK
from .base import Tool, ToolContext

# (query, allowed_domains, recency=None) -> list of {"title","url","snippet"}
# `recency` is an optional Tavily time_range ("day"/"week"/"month"/"year"): the default
# web_search backend ignores it (stays untimed), the recency backend applies it.
SearchFn = Callable[..., Awaitable[list[dict]]]

# Tavily's time_range accepts exactly these; anything else falls back to "year".
_RECENCY_WINDOWS = ("day", "week", "month", "year")
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


# Server-side query normalization, the layer where Gemini and ChatGPT put query understanding
# (their search backends rescue lazy queries; Tavily does not). Question scaffolding and errand
# verbs mislead lexical match: audited live, "what to prepare for tomorrow wc game" returned a
# gardening workshop because "prepare" matched. Civic action verbs (apply, appeal, renew,
# report) are content and are never stripped.
# ponytail: English scaffolding only; add Spanish scaffolding when a measured non-English
# retrieval failure shows the need.
_QUERY_PREFIX_RE = re.compile(
    r"^\s*(?:what(?:'s| is| are| should i| to| do i)?|how (?:do|can|should) i|how to|"
    r"can you|could you|please|tell me|show me|find me|give me|"
    r"i (?:need|want)(?: to)?|where (?:is|are|can i)|when (?:is|are|does)|"
    r"is there|are there|the|a|an)\b[\s,:]*",
    re.IGNORECASE,
)
_QUERY_ERRAND_RE = re.compile(
    r"\b(?:prepare(?: for)?|get ready(?: for)?|ready for|bring(?: to)?|wear(?: to)?|"
    r"pack(?: for)?)\b",
    re.IGNORECASE,
)


def _rewrite_query(query: str) -> str:
    """Normalize a query to the noun phrases and dates lexical search matches on.

    Applies to EVERY caller through the shared handler: the model's tool calls, the events
    module's internal lanes, and any future module. Falls back to the original whenever
    stripping guts the query, so a short or already search-shaped query passes untouched."""
    text = query.strip()
    for _ in range(6):
        stripped = _QUERY_PREFIX_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = _QUERY_ERRAND_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.?!")
    if len(text) < 6 or len(text.split()) < 2:
        return query.strip()
    return text


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
    recency: Optional[str] = None,
) -> list[dict]:
    """Local search fallback for a Tavily plan-limit response."""
    from ddgs import DDGS

    timelimit = {"day": "d", "week": "w", "month": "m", "year": "y"}.get(recency)
    try:
        results = await asyncio.to_thread(
            DDGS().text,
            query,
            max_results=20,
            timelimit=timelimit,
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
    ][:5]


async def _search_with_fallback(
    query: str,
    allowed_domains: list[str],
    *,
    recency: Optional[str] = None,
    **tavily_options,
) -> list[dict]:
    try:
        return await _tavily(query, allowed_domains, **tavily_options)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 432:
            raise
        return await _duckduckgo(query, allowed_domains, recency=recency)


async def tavily_search(query: str, allowed_domains: list[str], recency: Optional[str] = None) -> list[dict]:
    """Default backend, plain allowlisted search, no recency bias. IGNORES `recency` so the
    default web_search stays untimed and can serve general/historical/older-than-a-year queries."""
    return await _search_with_fallback(query, allowed_domains)


async def tavily_search_recent(query: str, allowed_domains: list[str], recency: Optional[str] = None) -> list[dict]:
    """Recency backend for the currency check. Applies a time window to bias toward recent items,
    but keeps Tavily's general (relevance-first) topic rather than the `news` topic. The `news`
    topic ranks by wire/trending volume, which buries a specific, slightly older LOCAL ruling under
    fresh NATIONAL headlines: measured against this allowlist it never surfaced the March-2026
    source-of-income appellate ruling for ANY query wording (broad, entity-rich, or by case name),
    while the relevance-first topic returned that exact ruling as the top on-point result. Recency
    comes from the entity-rich, year-bearing query the agent builds (rule 9) plus this window, not
    from a news-trending sort that trades away relevance.

    The window is agent-settable via `recency` (day/week/month/year): narrow to day/week for
    fast-moving current events, default to a full year for slow-moving rules/laws/rulings. Defaults
    to "year" when unset; defense in depth, any unexpected value also falls back to "year"."""
    window = recency if recency in _RECENCY_WINDOWS else "year"
    return await _search_with_fallback(
        query,
        allowed_domains,
        recency=window,
        time_range=window,
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
        raw_query = str(args["query"]).strip()
        query = _rewrite_query(raw_query)
        # `recency` only exists on the recent_developments schema; web_search never sets it (None),
        # and its untimed backend ignores it regardless.
        results = await search(query, domains, recency=args.get("recency"))
        results = [r for r in results if r.get("url")]
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
        # Mirror the vendors' exposed search queries: when the query was rewritten, say so,
        # so the model can refine its next search instead of re-sending the same sentence.
        header = f'Searched as: "{query}".\n\n' if query != raw_query else ""
        guidance = (
            "\n\nYou may cite only claims directly supported by an official excerpt. "
            "For details beyond an excerpt, call official_sources with its URL and a focused query. "
            "Editorial, news, community, and archived results remain discovery only."
            if any(tier == "authoritative" for _result, tier in tagged)
            else ""
        )
        return header + "\n\n".join(blocks) + guidance

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
    # Injected fakes (tests) drive both tools; production wires the recency-biased backend in.
    recent_search = search_fn or tavily_search_recent
    # The recency check unions the trusted allowlist with the subordinate news tier.
    recent_domains = sorted(set(allowlist) | {d.lower() for d in news_tier})

    web_search = _make_handler(
        search, allowlist, source_tiers, [],  # default search never tags/sees news
        abstain_msg=(
            "No results from trusted NYC sources for that query. "
            "Tell the user you couldn't find it on official sources rather than guessing."
        ),
    )
    recent_developments = _make_handler(
        recent_search, recent_domains, source_tiers, news_tier,
        abstain_msg=(
            "No recent developments found in trusted news or official sources for that query. "
            "Don't invent one, it's fine to say there's nothing new you can confirm."
        ),
    )

    prefer_param = {
        "prefer": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional domains to rank first (e.g. a topic's official sites).",
        },
    }
    # recent_developments ONLY, lets the agent pick the recency window per question.
    recency_param = {
        "recency": {
            "type": "string",
            "enum": list(_RECENCY_WINDOWS),
            "description": (
                "Optional recency window. Narrow to 'day' or 'week' for fast-moving current "
                "events (a breaking ruling, a just-announced change) so you don't pull stale "
                "year-old items. Leave it unset (defaults to 'year') for slow-moving rules, "
                "laws, or rulings where a year-wide window is appropriate."
            ),
        },
    }
    return [
        Tool(
            name="web_search",
            description=(
                "Search the live web for fresh or long-tail information and identity resolution. "
                "This tool is always available, including for current events and details that a "
                "structured NYC data tool may not cover. Known sources rank by trust; unlisted "
                "sources remain visible as unverified leads. "
                "Your ORIENTATION tool: when a resident reference is ambiguous or abbreviated, call "
                "this FIRST with a short noun-phrase query (the reference plus at most a date or "
                "NYC, never their whole sentence) to identify what they mean before choosing other "
                "tools. "
                "For the same missing fact, make one focused search and, when an authoritative result "
                "needs its page checked, one `official_sources` call. If that still does not support "
                "the fact, say you could not confirm it instead of issuing another search. "
                "Results are tagged authoritative, editorial, news, community, or unverified. "
                "Treat community and unverified results as leads and tell the user to verify. "
                "Pass `prefer` to boost the active topic's official "
                "domains. Cite every result; if nothing comes back, abstain."
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}, **prefer_param},
                "required": ["query"],
            },
            open_world=True,  # hits the open web (allowlisted)
            handler=web_search,
        ),
        Tool(
            name="recent_developments",
            description=(
                "The CURRENCY CHECK. Use AFTER you've grounded the authoritative answer in official "
                "sources, for legal / policy / benefits-rules / rights questions whose answer could "
                "have CHANGED recently (a court ruling, a new or amended law, an eligibility change). "
                "Build a SPECIFIC, entity-rich `query` from the actual rule/program/parties in the "
                "question, name the statute, program, or case and add 'ruling'/'law'/the year, e.g. "
                "'NYC Section 8 source of income discrimination court ruling 2026', NOT a broad "
                "'Section 8 news'. A broad query returns unrelated trending headlines instead of the "
                "on-point change. Searches the trusted allowlist PLUS a small curated set of reputable "
                "news + legal-news sources. News results are tagged '📰 news' and rank BELOW official "
                "sources, they are DEVELOPING/CONTESTED. RELEVANCE GATE: only surface a result if it "
                "bears on the SAME rule/law/program the user asked about. If what comes back is merely "
                "tangential (e.g. unrelated funding cuts when the question was about the discrimination "
                "law), STAY SILENT, saying nothing beats appending an off-topic caveat. When a result "
                "IS on point, surface it as a clearly labeled, DATED, CITED heads-up (e.g. 'Heads up, "
                "this may be changing: <X>, per <source> (<date>)') that NEVER overrides the official "
                "answer. CONTESTED LEGAL MATTER: if the development is a court ruling or a legal "
                "challenge to a right/benefit/protection, do NOT restate the ruling's court, holding, or "
                "scope from a news snippet as fact, and NEVER tell the user their protection is 'struck "
                "down / gone / annulled / no longer applies / may have changed.' LEAD with the protection "
                "that CURRENTLY STANDS (grounded + cited to the official source), then frame the "
                "litigation only as 'there is an active legal challenge, this could change, confirm the "
                "current status with 311 or the official agency.' Never name the court or characterize "
                "the outcome or scope; never imply a valid right is already gone. This holds in every "
                "language. Cite every result; if nothing on point comes back, don't invent a development."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    **prefer_param,
                    **recency_param,
                },
                "required": ["query"],
            },
            open_world=True,  # hits the open web (allowlist + curated news tier)
            handler=recent_developments,
        ),
    ]
