"""events module tool: `whats_on_events`, a thin composition (§10.3) of shared infra.

Merges the Ticketmaster Discovery client (structured backbone) + NYC Parks public
events (query_dataset over w3wp-dpdi) into one Event shape, filtered to future dates.
No hallucinated events: every row is grounded in a live source and cited with a link.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

from heynyc.core.index.corpus import clean_html, fetch_clean
from heynyc.core.ticketmaster import ticketmaster_events
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import query_dataset
from heynyc.core.tools.notify_nyc import _safe_fromstring
from heynyc.core.tools.official_sources import _relevant_chunks

PARKS_DATASET_ID = "w3wp-dpdi"  # NYC Parks Public Events (clean, upcoming, free/park-focused)
PARKS_SOURCE_URL = "https://www.nycgovparks.org/events"
NYC_TZ = ZoneInfo("America/New_York")
_SOURCE_TIMEOUT_S = 8.0
SECRET_NYC_WEEKEND_URL = "https://secretnyc.co/what-to-do-this-weekend-nyc/"
NYC_FOR_FREE_RSS_URL = "https://www.nycforfree.co/events?format=rss"


@dataclass
class Event:
    name: str
    start_date: str  # YYYY-MM-DD
    start_time: str  # local time string, possibly ""
    venue: str
    borough: str
    url: str
    source: str  # "Ticketmaster" | "NYC Parks"
    tier: str    # authoritative | editorial | community


def _iso_date(value: object) -> str:
    text = str(value or "")[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return ""
    return text


def _from_ticketmaster(raw: dict) -> Optional[Event]:
    dates = raw.get("dates") or {}
    if ((dates.get("status") or {}).get("code") or "").lower() in {
        "canceled", "cancelled", "postponed",
    }:
        return None
    start = dates.get("start") or {}
    start_date = _iso_date(start.get("localDate"))
    if not start_date:
        return None  # undated / TBA, don't surface as a real listing
    venues = (raw.get("_embedded") or {}).get("venues") or []
    venue = venues[0].get("name", "") if venues else ""
    borough = (venues[0].get("city") or {}).get("name", "") if venues else ""
    return Event(
        name=raw.get("name") or "", start_date=start_date,
        start_time=start.get("localTime") or "", venue=venue, borough=borough,
        url=raw.get("url", ""), source="Ticketmaster", tier="authoritative",
    )


def _from_parks(raw: dict) -> Optional[Event]:
    title = raw.get("title") or ""
    if title.lower().startswith(("cancelled", "canceled", "postponed")):
        return None
    start_date = _iso_date(raw.get("startdate"))
    if not start_date:
        return None
    link = raw.get("link")
    url = link.get("url", "") if isinstance(link, dict) else (link or "")
    return Event(
        name=title, start_date=start_date,
        start_time=raw.get("starttime") or "",
        venue=raw.get("parknames") or raw.get("location") or "",
        borough="", url=url or PARKS_SOURCE_URL, source="NYC Parks", tier="authoritative",
    )


def _future_only(events: list[Event], today: str) -> list[Event]:
    """Keep only events on/after `today` (ISO YYYY-MM-DD string compare is correct here)."""
    return [e for e in events if e.start_date >= today]


def _requested_window(query: str, today: str) -> tuple[str, str | None]:
    """Resolve only the relative window the tool can enforce without model interpretation."""
    current = date.fromisoformat(today)
    low = query.lower()
    if "this weekend" in low:
        start = (
            current + timedelta(days=5 - current.weekday())
            if current.weekday() < 5 else current
        )
        end = start + timedelta(days=6 - start.weekday())
        return start.isoformat(), end.isoformat()
    if "today" in low or "tonight" in low:
        return today, today
    if "this week" in low:
        return today, (current + timedelta(days=6 - current.weekday())).isoformat()
    return today, None


_NO_RESULTS = (
    "No upcoming NYC events matched that from the live sources (Ticketmaster + NYC Parks). "
    "Don't invent events, tell the user nothing grounded came up and suggest they check the "
    "official source directly."
)


def _event_block(ev: Event, cite: str) -> str:
    weekday = date.fromisoformat(ev.start_date).strftime("%A")
    when = f"{weekday}, {ev.start_date}" + (f" {ev.start_time}" if ev.start_time else "")
    where = f" @ {ev.venue}" if ev.venue else ""
    details = f"\n  Details: {ev.url}" if ev.url else ""
    return f"- {ev.name}{where}, {when} ({ev.source}) {{cite:{cite}}}{details}"


def _explicitly_free(events: list[Event], query: str) -> list[Event]:
    if "free" not in query.lower():
        return events
    return [event for event in events if re.search(r"\bfree\b", event.name, re.IGNORECASE)]


def _tonight_only(events: list[Event], now: datetime) -> list[Event]:
    cutoff = max(now.replace(tzinfo=None).time(), datetime.strptime("17:00", "%H:%M").time())
    kept = []
    for event in events:
        parsed = None
        for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(event.start_time.strip(), fmt).time()
                break
            except ValueError:
                continue
        if event.start_date == now.date().isoformat() and parsed is not None and parsed >= cutoff:
            kept.append(event)
    return kept


def _broad_temporal_query(query: str) -> bool:
    low = query.lower()
    event_intent = any(term in low for term in ("event", "what's on", "whats on", "happening", "things to do"))
    time_intent = any(term in low for term in ("today", "tonight", "weekend", "this week"))
    return event_intent and time_intent


def _editorial_query(query: str, window_start: str, window_end: Optional[str]) -> str:
    dates = [date.fromisoformat(window_start).strftime("%B %-d, %Y")]
    if window_end and window_end != window_start:
        dates.append(date.fromisoformat(window_end).strftime("%B %-d, %Y"))
    return f"{query} {' '.join(dates)}"


def _windowed_context(text: str, window_start: str, window_end: Optional[str]) -> str:
    start = date.fromisoformat(window_start)
    end = date.fromisoformat(window_end or window_start)
    patterns = []
    current = start
    while current <= end:
        patterns.extend((
            re.compile(rf"\b{re.escape(current.isoformat())}\b", re.IGNORECASE),
            re.compile(
                rf"\b{current.strftime('%B')}\s+{current.day}(?:st|nd|rd|th)?"
                rf"(?:,?\s+{current.year})?\b",
                re.IGNORECASE,
            ),
        ))
        current += timedelta(days=1)
    return "\n\n".join(
        block for block in text.split("\n\n")
        if any(pattern.search(block) for pattern in patterns)
    )


def _nyc_for_free_items(
    rss_text: str, window_start: str, window_end: Optional[str],
) -> tuple[str, list[tuple[str, str, str]]]:
    try:
        root = _safe_fromstring(rss_text)
    except Exception:
        return "", []
    dates = [date.fromisoformat(window_start)]
    if window_end and window_end != window_start:
        dates.append(date.fromisoformat(window_end))
    patterns = [
        re.compile(
            rf"\b{day.strftime('%B')}\s+{day.day}(?:st|nd|rd|th)?"
            rf"(?:,?\s+{day.year})?\b",
            re.IGNORECASE,
        )
        for day in dates
    ]
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        _unused, body = clean_html(item.findtext("description") or "")
        matched = next(
            (match.group(0) for pattern in patterns if (match := pattern.search(f"{title} {body}"))),
            "",
        )
        if title and url and matched:
            items.append((title, url, matched))
    return (root.findtext(".//lastBuildDate") or "").strip(), items


def _context_tools(ctx: ToolContext) -> tuple[Tool, ...]:
    from heynyc.core.tools.web_search import web_search_tools

    tools = []
    if ctx.toolbox and "index_search" in ctx.toolbox:
        tools.append(ctx.toolbox["index_search"])
    tiers = ctx.registry.source_tiers()
    trusted = [domain for domain, (tier, _) in tiers.items() if tier in {"authoritative", "editorial"}]
    tools.extend(web_search_tools(trusted, tiers, ctx.registry.news_tier()))
    return tuple(tools)


async def _editorial_context(
    ctx: ToolContext, window_start: str, window_end: Optional[str],
) -> str:
    own_client = ctx.http is None
    client = ctx.http or httpx.AsyncClient(timeout=_SOURCE_TIMEOUT_S)

    async def fetch_rss():
        response = await client.get(
            NYC_FOR_FREE_RSS_URL,
            follow_redirects=True,
            headers={"User-Agent": "HeyNYC/0.1"},
        )
        response.raise_for_status()
        return response.text

    try:
        secret, free_rss = await asyncio.gather(
            fetch_clean(SECRET_NYC_WEEKEND_URL, client),
            fetch_rss(),
            return_exceptions=True,
        )
    finally:
        if own_client:
            await client.aclose()

    blocks: list[str] = []
    if not isinstance(secret, BaseException):
        title, text = secret
        chunks = _relevant_chunks(
            f"{title} {text}", _editorial_query(ctx.query, window_start, window_end), limit=1,
        )
        if chunks:
            cite = ctx.citations.register(
                SECRET_NYC_WEEKEND_URL, snippet=chunks[0],
                title=title or "Secret NYC weekend guide", kind="WEB",
            )
            blocks.append(
                f"[{cite}] (editorial discovery, fetched live; publication freshness not "
                "independently verified; confirm details) "
                f"{title or 'Secret NYC weekend guide'} ({SECRET_NYC_WEEKEND_URL})\n{chunks[0]}"
            )
    if not isinstance(free_rss, BaseException):
        built, items = _nyc_for_free_items(free_rss, window_start, window_end)
        for title, url, matched_date in items[:3]:
            snippet = f"{title}. This event page explicitly mentions {matched_date}."
            cite = ctx.citations.register(
                url, snippet=snippet, title=title, kind="WEB", valid_as_of=built,
            )
            blocks.append(
                f"[{cite}] (editorial discovery, confirm details) {title} ({url})\n{snippet}"
            )
    return "\n\n".join(blocks) or "Current editorial event guides unavailable for this lookup."


async def _handler(args: dict, ctx: ToolContext) -> str:
    keyword = (args.get("keyword") or "").strip() or None
    classification = (args.get("classification") or "").strip() or None
    borough = (args.get("borough") or "").strip().lower()
    limit = int(args.get("limit") or 12)

    now = datetime.now(NYC_TZ)
    today = now.strftime("%Y-%m-%d")
    window_start, window_end = _requested_window(ctx.query, today)
    start_dt = (
        f"{window_start}T00:00:00Z"
        if window_start != today
        else now.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    async def ticketmaster_source():
        return await ticketmaster_events(
            keyword=keyword, classification=classification, start_datetime=start_dt,
            size=20, client=ctx.http,
        )

    async def parks_source():
        where = f"startdate >= '{window_start}'"
        if window_end:
            where += f" AND startdate <= '{window_end}T23:59:59'"
        return await query_dataset(
            PARKS_DATASET_ID, where=where, order="startdate",
            q=keyword, limit=50, client=ctx.http,
        )

    broad_context = _broad_temporal_query(ctx.query)
    sources = [
        ("ticketmaster", ticketmaster_source()),
        ("parks", parks_source()),
    ]
    if broad_context:
        sources.append(("editorial_guides", _editorial_context(ctx, window_start, window_end)))
        for tool in _context_tools(ctx):
            if tool.name == "index_search":
                topics = [module for module in ctx.registry.modules if module.parent == "events"]
                topic_queries = [
                    (
                        f"index_search:{topic.name}",
                        " ".join([
                            topic.name.replace("_", " "),
                            *(urlparse(seed).path.replace("-", " ").replace("/", " ")
                              for seed in topic.seeds[:1]),
                        ]),
                    )
                    for topic in topics
                ] or [("index_search", ctx.query)]
                sources.extend(
                    (name, tool.handler({"query": query}, ctx)) for name, query in topic_queries
                )
            else:
                if tool.name in {"web_search", "recent_developments"}:
                    editorial = [
                        domain for domain, (tier, _) in ctx.registry.source_tiers().items()
                        if tier == "editorial"
                    ]
                    tool_args = {
                        "query": _editorial_query(ctx.query, window_start, window_end),
                        "prefer": editorial,
                    }
                    if tool.name == "recent_developments":
                        tool_args["recency"] = "week"
                else:
                    tool_args = {
                        "citywide_only": not bool(borough),
                        **({"near": borough} if borough else {}),
                    }
                sources.append((tool.name, tool.handler(tool_args, ctx)))
    gathered = await asyncio.gather(*(
        asyncio.wait_for(call, timeout=_SOURCE_TIMEOUT_S) for _, call in sources
    ), return_exceptions=True)
    results = dict(zip((name for name, _ in sources), gathered))
    raw_tm = [] if isinstance(results["ticketmaster"], BaseException) else results["ticketmaster"]
    raw_parks = [] if isinstance(results["parks"], BaseException) else results["parks"]
    events = [e for e in (_from_ticketmaster(r) for r in raw_tm) if e]
    events += [e for e in (_from_parks(r) for r in raw_parks) if e]

    events = _future_only(events, window_start)
    if window_end:
        events = [e for e in events if e.start_date <= window_end]
    events = _explicitly_free(events, ctx.query)
    if "tonight" in ctx.query.lower():
        events = _tonight_only(events, now)
    if borough:
        events = [e for e in events if borough in e.borough.lower()] or events
    events.sort(key=lambda e: e.start_date)
    events = events[:limit]

    if not events and not broad_context:
        return _NO_RESULTS

    blocks = []
    for ev in events:
        weekday = date.fromisoformat(ev.start_date).strftime("%A")
        cite = ctx.citations.register(
            ev.url or PARKS_SOURCE_URL,
            snippet=f"{ev.name}, {weekday}, {ev.start_date} {ev.venue}".strip(),
            title=ev.name or "NYC event", kind="DATA", valid_as_of=ev.start_date,
        )
        blocks.append(_event_block(ev, cite))

    window = f" for {window_start} through {window_end}" if window_end else ""
    free_scope = " whose official source titles explicitly say free" if "free" in ctx.query.lower() else ""
    header = (
        f"Upcoming NYC events{window}{free_scope} from live sources (Ticketmaster + NYC Parks). "
        "Each links to its official page, cite them and don't add events that aren't listed here:\n"
    )
    catalog = header + "\n".join(blocks) if blocks else _NO_RESULTS
    if not broad_context:
        return catalog

    index_results = [value for name, value in results.items() if name.startswith("index_search")]
    index_contexts = [value for value in index_results if not isinstance(value, BaseException)]
    index_context = "\n\n".join(index_contexts) or "Curated official event context unavailable for this lookup."
    web_context = results.get("web_search", "Official web context unavailable for this lookup.")
    if isinstance(web_context, BaseException):
        web_context = "Official web context unavailable for this lookup."
    else:
        web_context = _windowed_context(str(web_context), window_start, window_end) or (
            "Official web context had no result explicitly dated for this window."
        )
    recent_context = results.get(
        "recent_developments", "Current editorial and news context unavailable for this lookup.",
    )
    if isinstance(recent_context, BaseException):
        recent_context = "Current editorial and news context unavailable for this lookup."
    else:
        recent_context = _windowed_context(str(recent_context), window_start, window_end) or (
            "Current editorial and news context had no result explicitly dated for this window."
        )
    editorial_context = results.get(
        "editorial_guides", "Current editorial event guides unavailable for this lookup.",
    )
    if isinstance(editorial_context, BaseException):
        editorial_context = "Current editorial event guides unavailable for this lookup."
    return (
        f"{catalog}\n\n"
        "Newly retrieved current city context for this event-planning question:\n"
        f"Curated official and seasonal context:\n{index_context}\n\n"
        f"Official event and seasonal context:\n{web_context}\n\n"
        f"Current editorial event guides:\n{editorial_context}\n\n"
        f"Current editorial and news discovery:\n{recent_context}\n\n"
        "Synthesis rules: supplement the structured catalog with relevant official web results. "
        "For every recommended event, copy its direct source URL beside the item and include any "
        "known date, time, place, and ticket or reservation step. Prefer individual event pages "
        "over roundup pages. Voice and format: start with one warm sentence about what stands out; "
        "give at most 5 options by default, each with a short bold title and one light emoji; keep "
        "the logistics compact; briefly group the rest by category or time of day and offer to list "
        "them; merge sources that describe the same event into one option; mention any today-only "
        "advisory once after the event list; finish with one natural "
        "narrowing question. Do not call an event "
        "free unless its cited evidence establishes that. Keep "
        "prior-conversation facts separate from these newly retrieved results."
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="whats_on_events",
            description=(
                "Find upcoming NYC events (concerts, sports, festivals, free park events, watch "
                "parties) from live Ticketmaster, NYC Parks, current editorial guides, and trusted "
                "web sources. Pass `keyword` (e.g. "
                "'world cup', 'jazz'), optional `classification` (Music/Sports/Arts & Theatre), "
                "and optional `borough`. Returns grounded, dated, linked listings, future events "
                "only. Use this for 'what's happening' / 'events this weekend' questions; it never "
                "invents events and already coordinates the event retrieval lanes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Topic/keyword, e.g. 'world cup', 'jazz', 'free'."},
                    "classification": {"type": "string", "description": "Optional Ticketmaster segment: Music, Sports, Arts & Theatre, etc."},
                    "borough": {"type": "string", "description": "Optional borough/city filter, e.g. 'Brooklyn'."},
                    "limit": {"type": "integer", "description": "Max events to return (default 12)."},
                },
            },
            handler=_handler,
            open_world=True,  # hits live Ticketmaster + Socrata
        )
    ]
