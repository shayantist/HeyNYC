"""events module tool: `whats_on_events`, a thin composition (§10.3) of shared infra.

Merges the Ticketmaster Discovery client (structured backbone) + NYC Parks public
events (query_dataset over w3wp-dpdi) into one Event shape, filtered to future dates.
No hallucinated events: every row is grounded in a live source and cited with a link.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from heynyc.core.ticketmaster import ticketmaster_events
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import query_dataset

PARKS_DATASET_ID = "w3wp-dpdi"  # NYC Parks Public Events (clean, upcoming, free/park-focused)
PARKS_SOURCE_URL = "https://www.nycgovparks.org/events"


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


def _from_ticketmaster(raw: dict) -> Optional[Event]:
    start = (raw.get("dates") or {}).get("start") or {}
    start_date = start.get("localDate") or ""
    if not start_date:
        return None  # undated / TBA, don't surface as a real listing
    venues = (raw.get("_embedded") or {}).get("venues") or []
    venue = venues[0].get("name", "") if venues else ""
    borough = (venues[0].get("city") or {}).get("name", "") if venues else ""
    return Event(
        name=raw.get("name", ""), start_date=start_date,
        start_time=start.get("localTime") or "", venue=venue, borough=borough,
        url=raw.get("url", ""), source="Ticketmaster", tier="authoritative",
    )


def _from_parks(raw: dict) -> Optional[Event]:
    start_date = (raw.get("startdate") or "")[:10]  # "2026-06-17T00:00:00.000" -> "2026-06-17"
    if not start_date:
        return None
    link = raw.get("link")
    url = link.get("url", "") if isinstance(link, dict) else (link or "")
    return Event(
        name=raw.get("title", ""), start_date=start_date,
        start_time=raw.get("starttime") or "",
        venue=raw.get("parknames") or raw.get("location") or "",
        borough="", url=url or PARKS_SOURCE_URL, source="NYC Parks", tier="authoritative",
    )


def _future_only(events: list[Event], today: str) -> list[Event]:
    """Keep only events on/after `today` (ISO YYYY-MM-DD string compare is correct here)."""
    return [e for e in events if e.start_date >= today]


_NO_RESULTS = (
    "No upcoming NYC events matched that from the live sources (Ticketmaster + NYC Parks). "
    "Don't invent events, tell the user nothing grounded came up and suggest they check the "
    "official source directly."
)


def _event_block(ev: Event, cite: str) -> str:
    when = ev.start_date + (f" {ev.start_time}" if ev.start_time else "")
    where = f" @ {ev.venue}" if ev.venue else ""
    return f"- {ev.name}{where}, {when} ({ev.source}) {{cite:{cite}}}"


async def _handler(args: dict, ctx: ToolContext) -> str:
    keyword = (args.get("keyword") or "").strip() or None
    classification = (args.get("classification") or "").strip() or None
    borough = (args.get("borough") or "").strip().lower()
    limit = int(args.get("limit") or 12)

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    start_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    events: list[Event] = []
    # Layer 1a: Ticketmaster (structured backbone). Failures degrade, never crash.
    try:
        raw_tm = await ticketmaster_events(
            keyword=keyword, classification=classification, start_datetime=start_dt,
            size=20, client=ctx.http,
        )
        events += [e for e in (_from_ticketmaster(r) for r in raw_tm) if e]
    except httpx.HTTPError:
        pass
    # Layer 1b: NYC Parks public events (free/park supplement), date-filtered server-side.
    try:
        raw_parks = await query_dataset(
            PARKS_DATASET_ID, where=f"startdate >= '{today}'", order="startdate",
            q=keyword, limit=50, client=ctx.http,
        )
        events += [e for e in (_from_parks(r) for r in raw_parks) if e]
    except httpx.HTTPError:
        pass

    events = _future_only(events, today)
    if borough:
        events = [e for e in events if borough in e.borough.lower()] or events
    events.sort(key=lambda e: e.start_date)
    events = events[:limit]

    if not events:
        return _NO_RESULTS

    blocks = []
    for ev in events:
        cite = ctx.citations.register(
            ev.url or PARKS_SOURCE_URL,
            snippet=f"{ev.name}, {ev.start_date} {ev.venue}".strip(),
            title=ev.name or "NYC event", kind="DATA", valid_as_of=ev.start_date,
        )
        blocks.append(_event_block(ev, cite))

    header = (
        "Upcoming NYC events from live sources (Ticketmaster + NYC Parks). Each links to its "
        "official page, cite them and don't add events that aren't listed here:\n"
    )
    return header + "\n".join(blocks)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="whats_on_events",
            description=(
                "Find upcoming NYC events (concerts, sports, festivals, free park events, watch "
                "parties) from live sources, Ticketmaster + NYC Parks. Pass `keyword` (e.g. "
                "'world cup', 'jazz'), optional `classification` (Music/Sports/Arts & Theatre), "
                "and optional `borough`. Returns grounded, dated, linked listings, future events "
                "only. Use this for 'what's happening' / 'events this weekend' questions; it never "
                "invents events. For curated topic pages use index_search; for the long tail use "
                "web_search."
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
