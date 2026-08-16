"""events module tool: `find_nyc_events`, a thin composition (§10.3) of shared infra.

Merges the Ticketmaster Discovery client (structured backbone) + NYC Parks public
events (query_dataset over w3wp-dpdi) into one Event shape, filtered to future dates.
No hallucinated events: every row is grounded in a live source and cited with a link.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.index.corpus import clean_html
from heynyc.core.ticketmaster import (
    DISCOVERY_URL,
    TicketmasterSearchResult,
    ticketmaster_events,
)
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import (
    dataset_url,
    query_dataset,
    query_dataset_pages,
    row_url,
)

PARKS_DATASET_ID = "w3wp-dpdi"  # NYC Parks Public Events (clean, upcoming, free/park-focused)
PARKS_SOURCE_URL = "https://www.nycgovparks.org/events"
# NYC Permitted Event Information: the public street life (street fairs, farmers markets, block
# parties, parades, plaza events) the Parks/Ticketmaster lanes structurally miss. The bulk of the
# dataset is sport field/court reservations (agency "Parks Department"); the street events we want
# are the "Street Activity Permit Office" agency slice, minus film/TV production load-ins.
PERMITTED_DATASET_ID = "tvpp-9vvx"
PERMITTED_AGENCY = "Street Activity Permit Office"
PERMITTED_SOURCE_URL = (
    "https://data.cityofnewyork.us/City-Government/NYC-Permitted-Event-Information/tvpp-9vvx"
)
NYC_TZ = ZoneInfo("America/New_York")
_SOURCE_TIMEOUT_S = 8.0
_PARK_BOROUGHS = {
    "B": "Brooklyn",
    "M": "Manhattan",
    "Q": "Queens",
    "R": "Staten Island",
    "X": "Bronx",
}
_TICKETMASTER_NYC_CITIES = {
    "bronx", "brooklyn", "new york", "new york city", "queens", "staten island",
}


@dataclass
class Event:
    name: str
    start_date: str  # YYYY-MM-DD
    start_time: str  # local time string, possibly ""
    venue: str
    borough: str
    url: str
    source: str  # "Ticketmaster Discovery" | "NYC Parks" | "NYC Permitted Events"
    tier: str    # authoritative | editorial | community
    free_evidence: str = ""
    audience: str = ""
    end_time: str = ""
    provider_status: str = ""
    timezone: str = ""
    public_sale_start: str = ""
    public_sale_end: str = ""
    public_sale_start_tbd: bool | None = None
    accessibility_info: str = ""
    venue_accessibility: str = ""
    publishing_source: str = ""
    provider_id: str = ""
    provider_record: dict = field(default_factory=dict)
    retrieved_at: str = ""


def _iso_date(value: object) -> str:
    text = str(value or "")[:10]
    try:
        date.fromisoformat(text)
    except ValueError:
        return ""
    return text


def _parks_time(value: object) -> str:
    text = str(value or "").strip()
    if " " not in text:
        return text
    try:
        parsed = datetime.strptime(text.rsplit(" ", 1)[-1], "%H:%M:%S")
    except ValueError:
        return text
    return parsed.strftime("%I:%M %p").lstrip("0")


def _from_ticketmaster(raw: dict, *, retrieved_at: str = "") -> Optional[Event]:
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
    if borough and borough.strip().casefold() not in _TICKETMASTER_NYC_CITIES:
        return None
    public_sale = (raw.get("sales") or {}).get("public") or {}
    start_tbd = public_sale.get("startTBD")
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    return Event(
        name=raw.get("name") or "", start_date=start_date,
        start_time=start.get("localTime") or "", venue=venue, borough=borough,
        url=raw.get("url", ""), source="Ticketmaster Discovery", tier="authoritative",
        provider_status=((dates.get("status") or {}).get("code") or "").lower(),
        timezone=dates.get("timezone") or "",
        public_sale_start=public_sale.get("startDateTime") or "",
        public_sale_end=public_sale.get("endDateTime") or "",
        public_sale_start_tbd=start_tbd if isinstance(start_tbd, bool) else None,
        accessibility_info=(raw.get("accessibility") or {}).get("info") or "",
        venue_accessibility=(venues[0].get("accessibleSeatingDetail") or "") if venues else "",
        publishing_source=source.get("name") or source.get("id") or "",
        provider_id=str(raw.get("id") or raw.get("url") or ""),
        provider_record=raw,
        retrieved_at=retrieved_at,
    )


def _from_parks(raw: dict, *, retrieved_at: str = "") -> Optional[Event]:
    title = raw.get("title") or ""
    if title.lower().startswith(("cancelled", "canceled", "postponed")):
        return None
    start_date = _iso_date(raw.get("startdate"))
    if not start_date:
        return None
    link = raw.get("link")
    url = link.get("url", "") if isinstance(link, dict) else (link or "")
    if url.startswith("http://www.nycgovparks.org/"):
        url = f"https://{url.removeprefix('http://')}"
    _unused, description = clean_html(str(raw.get("description") or ""))
    free_evidence = next(
        (
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", description)
            if re.search(r"\bfree\b", sentence, re.IGNORECASE)
        ),
        "",
    )
    categories = str(raw.get("categories") or "")
    audience = next(
        (category.strip() for category in categories.split("|")
         if category.strip().lower().startswith("best for ")),
        "",
    )
    park_id = str(raw.get("parkids") or "").strip().upper()
    return Event(
        name=title, start_date=start_date,
        start_time=_parks_time(raw.get("starttime")),
        venue=raw.get("parknames") or raw.get("location") or "",
        borough=_PARK_BOROUGHS.get(park_id[:1], ""),
        url=url or PARKS_SOURCE_URL, source="NYC Parks", tier="authoritative",
        free_evidence=free_evidence, audience=audience,
        provider_id=str(raw.get(":id") or url or title),
        provider_record=raw,
        retrieved_at=retrieved_at,
    )


def _from_permitted(raw: dict, *, retrieved_at: str = "") -> Optional[Event]:
    name = raw.get("event_name") or ""
    # No status column exists on this permit dataset, so mirror the Parks/Ticketmaster
    # cancellation discipline off the name: a permit marked cancelled/postponed is not happening.
    if name.lower().startswith(("cancelled", "canceled", "postponed")):
        return None
    start_date = _iso_date(raw.get("start_date_time"))
    if not start_date:
        return None
    raw_start = str(raw.get("start_date_time") or "")
    start_time = raw_start[11:16] if len(raw_start) >= 16 else ""  # "HH:MM" from the ISO stamp
    raw_end = str(raw.get("end_date_time") or "")
    end_time = raw_end[11:16] if len(raw_end) >= 16 else ""
    row_id = str(raw.get(":id") or "")
    return Event(
        name=name, start_date=start_date, start_time=start_time,
        venue=raw.get("event_location") or "", borough=raw.get("event_borough") or "",
        url=row_url(PERMITTED_DATASET_ID, row_id) if row_id else PERMITTED_SOURCE_URL,
        source="NYC Permitted Events", tier="authoritative",
        end_time=end_time,
        provider_id=row_id or str(raw.get("event_id") or ""),
        provider_record=raw,
        retrieved_at=retrieved_at,
    )


def _future_only(events: list[Event], today: str) -> list[Event]:
    """Keep only events on/after `today` (ISO YYYY-MM-DD string compare is correct here)."""
    return [e for e in events if e.start_date >= today]


def _not_ended_today(events: list[Event], now: datetime) -> list[Event]:
    """Keep same-day rows only when their structured time shows they remain attendable."""
    current_time = now.replace(tzinfo=None).time()
    kept = []
    for event in events:
        if event.start_date != now.date().isoformat():
            kept.append(event)
            continue
        end = _parse_start_time(event.end_time)
        start = _parse_start_time(event.start_time)
        if (end is not None and end >= current_time) or (
            not event.end_time.strip()
            and start is not None
            and start >= current_time
        ):
            kept.append(event)
    return kept


def _shortlist(events: list[Event], limit: int) -> list[Event]:
    """Cap the merged lanes while preserving source and requested-date coverage."""
    unique: list[Event] = []
    identities: set[tuple[str, ...]] = set()
    for event in events:
        identity = (
            event.source.casefold(),
            event.start_date,
            event.start_time.casefold(),
            (
                event.venue.casefold()
                if event.venue and event.start_time
                else event.name.casefold()
            ),
        )
        if identity in identities:
            continue
        identities.add(identity)
        unique.append(event)
    seen: dict[tuple[str, str], int] = {}
    ranked: list[tuple[int, str, Event]] = []
    for event in sorted(unique, key=lambda e: e.start_date):
        group = (event.source, event.start_date)
        rank = seen.get(group, 0)
        seen[group] = rank + 1
        ranked.append((rank, event.start_date, event))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [event for _, _, event in ranked[:limit]]


def _resident_limit(query: str) -> tuple[int, bool]:
    match = re.search(
        r"\b(?:show|give|list|find|send)(?:\s+me)?\s+(?:up to\s+)?(\d{1,2})\b"
        r"|\b(\d{1,2})\s+(?:events?|options?|choices?)\b",
        query,
        re.IGNORECASE,
    )
    if not match:
        return 5, False
    return max(1, min(int(next(group for group in match.groups() if group)), 20)), True


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
    if re.search(r"\btom{1,2}or{1,2}ow(?:'?s)?\b|\btmrw\b|\btm\b|\bma[ñn]ana\b", low):
        day = (current + timedelta(days=1)).isoformat()
        return day, day
    numeric = re.search(r"\b(\d{1,2})/(\d{1,2})\b", low)
    if numeric:
        try:
            day_date = date(current.year, int(numeric.group(1)), int(numeric.group(2)))
        except ValueError:
            day_date = None
        if day_date is not None:
            if day_date < current:
                day_date = day_date.replace(year=current.year + 1)
            return day_date.isoformat(), day_date.isoformat()
    if "today" in low or "tonight" in low:
        return today, today
    if "this week" in low:
        return today, (current + timedelta(days=6 - current.weekday())).isoformat()
    return today, None


_NO_RESULTS = (
    "No upcoming NYC events matched that from the live sources (Ticketmaster + NYC Parks + "
    "NYC Permitted Events)."
)

_TIME_FORMATS = ("%I:%M %p", "%I:%M%p", "%H:%M:%S", "%H:%M")


def _parse_start_time(text: str):
    """Best-effort local wall-clock time from an event's free-form start_time, else None."""
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).time()
        except ValueError:
            continue
    return None


def _today_timing_note(ev: Event, now: datetime) -> str:
    """Describe a known same-day start relative to the NYC clock."""
    if ev.start_date != now.date().isoformat() or not ev.start_time.strip():
        return ""
    parsed = _parse_start_time(ev.start_time)
    if parsed is None:
        return ""
    current_time = now.replace(tzinfo=None).time()
    if parsed < current_time:
        end = _parse_start_time(ev.end_time)
        if end is not None and end >= current_time:
            return "; in progress"
        return "; already started or ended earlier today"
    return "; starts later today"


def _event_block(ev: Event, cite: str, now: Optional[datetime] = None) -> str:
    weekday = date.fromisoformat(ev.start_date).strftime("%A")
    when = f"{weekday}, {ev.start_date}" + (f" {ev.start_time}" if ev.start_time else "")
    where = f" @ {ev.venue}" if ev.venue else ""
    timing = _today_timing_note(ev, now) if now else ""
    end = f"; ends {ev.end_time}" if ev.end_time else ""
    free = "; free" if ev.free_evidence else ""
    audience = f"; {ev.audience}" if ev.audience else ""
    status = f"; status {ev.provider_status}" if ev.provider_status else ""
    accessibility = (
        f"; accessibility: {ev.accessibility_info or ev.venue_accessibility}"
        if ev.accessibility_info or ev.venue_accessibility
        else ""
    )
    details = f"\n  Details: {ev.url}" if ev.url else ""
    return (
        f"- {ev.name}{where}, {when}{timing}{end}{free}{audience}{status}{accessibility} "
        f"({ev.source}) {{cite:{cite}}}{details}"
    )


def _explicitly_free(events: list[Event], query: str) -> list[Event]:
    if "free" not in query.lower():
        return events
    return [
        event for event in events
        if event.free_evidence or re.search(r"\bfree\b", event.name, re.IGNORECASE)
    ]


_GENERIC_EVENT_WORDS = {
    "a", "activities", "activity", "any", "anything", "around", "city", "do",
    "event", "events", "find", "for", "free", "going", "happen", "happening",
    "in", "me", "new", "nyc", "on", "show", "stuff", "thing", "things", "this",
    "to", "today", "tonight", "week", "weekend", "what", "whats", "york",
}


def _keyword_terms(keyword: str) -> set[str]:
    raw_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", keyword.lower())
        if len(term) >= 3 and term not in _GENERIC_EVENT_WORDS
    }
    return {
        term[:-1] if len(term) > 4 and term.endswith("s") else term
        for term in raw_terms
    }


def _matches_keyword(event: Event, keyword: str) -> bool:
    terms = _keyword_terms(keyword)
    blob = " ".join((event.name, event.venue, event.audience, event.free_evidence)).lower()
    return not terms or any(term in blob for term in terms)


def _tonight_only(events: list[Event], now: datetime) -> list[Event]:
    cutoff = max(now.replace(tzinfo=None).time(), datetime.strptime("17:00", "%H:%M").time())
    kept = []
    for event in events:
        parsed = _parse_start_time(event.start_time)
        if event.start_date == now.date().isoformat() and parsed is not None and parsed >= cutoff:
            kept.append(event)
    return kept


async def _handler(args: dict, ctx: ToolContext) -> str:
    keyword = (args.get("keyword") or "").strip() or None
    if keyword and not _keyword_terms(keyword):
        keyword = None
    classification = (args.get("classification") or "").strip() or None
    borough = (args.get("borough") or "").strip().lower()
    audience = (args.get("audience") or "").strip().lower()
    web_query = (args.get("web_query") or "").strip()
    setting = (args.get("setting") or "").strip().lower()
    if audience not in {"", "kids"}:
        raise ValueError(f"Unsupported audience: {audience}")
    if setting not in {"", "indoor", "outdoor"}:
        raise ValueError(f"Unsupported setting: {setting}")
    limit, resident_requested_count = _resident_limit(ctx.query)

    now = datetime.now(NYC_TZ)
    today = now.strftime("%Y-%m-%d")

    def _valid_iso(value):
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError):
            return None

    # F085: the resident's timeframe is the model's to state (a date, a range, a month, the
    # past); the deterministic relative phrases remain the fallback when no args arrive.
    arg_start = _valid_iso((args.get("window_start") or "").strip() or None)
    arg_end = _valid_iso((args.get("window_end") or "").strip() or None)
    if arg_start or arg_end:
        window_start, window_end = arg_start or today, arg_end
    else:
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
        return await query_dataset_pages(
            PARKS_DATASET_ID, where=where, order="startdate",
            client=ctx.http, _query=query_dataset,
        )

    # Select the public street-event slice by the dataset's own agency/type FIELDS (never by
    # matching event names): the "Street Activity Permit Office" agency drops the ~6.5k/week Parks
    # sport reservations, and excluding "Production Event" drops the film/TV load-ins.
    # ponytail: window is start-date-in-window (mirrors the Parks lane). A season-long single
    # permit that started before the window (a summer-long farmers market) is missed; most rows
    # are per-occurrence so this catches ~89% of active street events. Add overlap logic only if
    # the season-permit miss proves material.
    permitted_where = (
        f"event_agency = '{PERMITTED_AGENCY}' AND event_type != 'Production Event' "
        f"AND start_date_time >= '{window_start}'"
        + (f" AND start_date_time <= '{window_end}T23:59:59'" if window_end else "")
    )

    async def permitted_source():
        return await query_dataset_pages(
            PERMITTED_DATASET_ID, where=permitted_where, order="start_date_time",
            client=ctx.http, _query=query_dataset,
        )
    catalog_sources = {
        "ticketmaster": ticketmaster_source,
        "parks": parks_source,
        "permitted": permitted_source,
    }
    catalog_labels = {
        "ticketmaster": "Ticketmaster",
        "parks": "NYC Parks",
        "permitted": "NYC Permitted Events",
    }
    sources = [(name, source()) for name, source in catalog_sources.items()]
    broad_web = (
        (bool(web_query) or (not keyword and not classification))
        and bool(ctx.query.strip())
        and ctx.toolbox is not None
        and "web_search" in ctx.toolbox
    )
    if broad_web:
        sources.append((
            "broad_web",
            ctx.toolbox["web_search"].handler(
                {"query": web_query or ctx.query.strip(), "count": 5}, ctx
            ),
        ))
    gathered = await asyncio.gather(*(
        asyncio.wait_for(call, timeout=_SOURCE_TIMEOUT_S) for _, call in sources
    ), return_exceptions=True)
    results = dict(zip((name for name, _ in sources), gathered))
    failed_catalog = [
        name for name in catalog_sources if isinstance(results[name], BaseException)
    ]
    retryable_failed = [
        name for name in failed_catalog
    ]
    if retryable_failed:
        retried = await asyncio.gather(*(
            asyncio.wait_for(catalog_sources[name](), timeout=_SOURCE_TIMEOUT_S)
            for name in retryable_failed
        ), return_exceptions=True)
        results.update(zip(retryable_failed, retried))
    unavailable_catalog = [
        name for name in catalog_sources if isinstance(results[name], BaseException)
    ]
    web_context = results.get("broad_web")
    web_failed = isinstance(web_context, BaseException)
    if web_failed:
        web_context = None
    web_appendix = (
        "\n\nCurrent web event leads, searched in parallel with the structured catalogs:\n"
        "Context only, not additional event choices. Use a current web lead only by "
        "replacing a structured choice, after verifying every resident constraint and keeping "
        "the complete answer within the requested result limit:\n"
        f"{web_context}"
        if web_context
        else (
            "\n\nCurrent web event leads were unavailable. Results are partial."
            if web_failed
            else ""
        )
    )
    ticketmaster_result = results["ticketmaster"]
    if not isinstance(ticketmaster_result, BaseException):
        if ticketmaster_result.status == "unavailable":
            unavailable_catalog.append("ticketmaster")
        raw_tm = ticketmaster_result.events
    else:
        raw_tm = []
    raw_parks = [] if isinstance(results["parks"], BaseException) else results["parks"].records
    raw_permitted = (
        [] if isinstance(results["permitted"], BaseException) else results["permitted"].records
    )
    retrieved_at = now.astimezone(timezone.utc).isoformat()
    partial_socrata = {
        name: results[name]
        for name in ("parks", "permitted")
        if not isinstance(results[name], BaseException) and not results[name].complete
    }
    events = [
        e for e in (
            _from_ticketmaster(r, retrieved_at=ticketmaster_result.retrieved_at)
            for r in raw_tm
        ) if e
    ]
    events += [e for e in (_from_parks(r, retrieved_at=retrieved_at) for r in raw_parks) if e]
    events += [
        e for e in (_from_permitted(r, retrieved_at=retrieved_at) for r in raw_permitted) if e
    ]
    if keyword:
        events = [
            event for event in events
            if event.source == "Ticketmaster Discovery"
            or _matches_keyword(event, keyword)
        ]

    def _window_filter(rows: list[Event]) -> list[Event]:
        kept = _future_only(rows, window_start)
        kept = _not_ended_today(kept, now)
        if window_end:
            kept = [e for e in kept if e.start_date <= window_end]
        kept = _explicitly_free(kept, ctx.query)
        if "tonight" in ctx.query.lower():
            kept = _tonight_only(kept, now)
        if borough:
            kept = [e for e in kept if borough in e.borough.lower()]
        if audience == "kids":
            kept = [e for e in kept if "kids" in e.audience.lower()]
        return _shortlist(kept, limit)

    events = _window_filter(events)
    limited_catalog = set(unavailable_catalog)
    coverage_note = ""
    if limited_catalog:
        names = ", ".join(
            catalog_labels[name] for name in catalog_sources if name in limited_catalog
        )
        coverage_note = (
            f"Sources unavailable for part of this lookup: {names}. Results are partial.\n"
        )
    for name, result in partial_socrata.items():
        dataset_id = PARKS_DATASET_ID if name == "parks" else PERMITTED_DATASET_ID
        snapshot = {
            "dataset_id": dataset_id,
            "complete": result.complete,
            "pages_fetched": result.pages_fetched,
            "returned_count": len(result.records),
            "next_offset": result.next_offset,
            "error": result.error,
            "retrieved_at": retrieved_at,
        }
        coverage_cite = ctx.citations.register(
            dataset_url(dataset_id),
            snippet=(
                f"{catalog_labels[name]} returned {len(result.records)} rows before "
                f"{result.error or 'an incomplete page'} at offset {result.next_offset}."
            ),
            title=f"{catalog_labels[name]} search coverage",
            kind="DATA",
            provenance=data_provenance(
                snapshot,
                record_id=f"{dataset_id}:query:{retrieved_at}",
                field_pointer="/",
            ),
        )
        coverage_note += (
            f"{catalog_labels[name]} returned partial results after {result.pages_fetched} page(s) "
            f"{{cite:{coverage_cite}}}; do not claim complete catalog coverage.\n"
        )
    if (
        isinstance(ticketmaster_result, TicketmasterSearchResult)
        and ticketmaster_result.status == "partial"
    ):
        returned = len(ticketmaster_result.events)
        total = ticketmaster_result.total_elements
        coverage_snapshot = {
            "status": ticketmaster_result.status,
            "page_number": ticketmaster_result.page_number,
            "page_size": ticketmaster_result.page_size,
            "returned": returned,
            "total_elements": total,
            "total_pages": ticketmaster_result.total_pages,
            "next_page": ticketmaster_result.next_page,
            "retrieved_at": ticketmaster_result.retrieved_at,
        }
        coverage_cite = ctx.citations.register(
            DISCOVERY_URL,
            snippet=(
                f"Ticketmaster returned page {ticketmaster_result.page_number or 0} with "
                f"{returned} of {total} listings."
                if total is not None
                else f"Ticketmaster returned one page with {returned} listings."
            ),
            title="Ticketmaster search coverage",
            kind="DATA",
            provenance=data_provenance(
                coverage_snapshot,
                record_id=(
                    f"ticketmaster:page:{ticketmaster_result.page_number or 0}:"
                    f"{ticketmaster_result.retrieved_at}"
                ),
                field_pointer="/",
            ),
        )
        coverage_note += (
            f"Ticketmaster returned one page ({returned} of {total} listings) "
            f"{{cite:{coverage_cite}}}; its catalog "
            "coverage is partial.\n"
            if total is not None
            else f"Ticketmaster returned one page without complete paging metadata "
            f"{{cite:{coverage_cite}}}; its catalog "
            "coverage is partial.\n"
        )
    no_results = (
        f"{coverage_note}No matching events were confirmed from the sources that responded."
        if coverage_note
        else _NO_RESULTS
    )
    setting_note = (
        f"Requested setting: {setting}. Structured catalog rows do not have a source-backed "
        "indoor or outdoor field, so do not infer indoor or outdoor from a venue name.\n"
        if setting
        else ""
    )
    if not events:
        parks_cite = ctx.citations.register(
            PARKS_SOURCE_URL,
            snippet="NYC Parks official events calendar",
            title="NYC Parks events",
            kind="WEB",
            valid_as_of=today,
        )
        permitted_cite = ctx.citations.register(
            PERMITTED_SOURCE_URL,
            snippet="NYC Permitted Events official public dataset",
            title="NYC Permitted Events",
            kind="WEB",
            valid_as_of=today,
        )
        no_results += (
            f" Official indexes the resident can check directly: NYC Parks {PARKS_SOURCE_URL} "
            f"{{cite:{parks_cite}}} and NYC Permitted Events {PERMITTED_SOURCE_URL} "
            f"{{cite:{permitted_cite}}}."
        )
    if not events:
        return setting_note + no_results + web_appendix

    blocks = []
    for ev in events:
        weekday = date.fromisoformat(ev.start_date).strftime("%A")
        # The snapshot carries the FULL row the model will describe, time and source included,
        # so cited prose stays supported by its own evidence.
        timing = _today_timing_note(ev, now).removeprefix("; ")
        snippet_bits = [
            f"Name: {ev.name}",
            f"Date: {weekday}, {ev.start_date}",
            f"Start time: {ev.start_time}" if ev.start_time else "",
            f"End time: {ev.end_time}" if ev.end_time else "",
            f"Venue: {ev.venue}" if ev.venue else "",
            f"Borough: {ev.borough}" if ev.borough else "",
            f"Timing at lookup: {timing}" if timing else "",
            f"Cost evidence: {ev.free_evidence}" if ev.free_evidence else "",
            f"Audience: {ev.audience}" if ev.audience else "",
            f"Ticketmaster status: {ev.provider_status}" if ev.provider_status else "",
            f"Timezone: {ev.timezone}" if ev.timezone else "",
            f"Public sale start: {ev.public_sale_start}" if ev.public_sale_start else "",
            f"Public sale end: {ev.public_sale_end}" if ev.public_sale_end else "",
            (
                f"Public sale start TBD: {ev.public_sale_start_tbd}"
                if ev.public_sale_start_tbd is not None
                else ""
            ),
            f"Accessibility: {ev.accessibility_info}" if ev.accessibility_info else "",
            (
                f"Venue accessibility: {ev.venue_accessibility}"
                if ev.venue_accessibility
                else ""
            ),
            f"Source: {ev.source}",
        ]
        provenance = data_provenance(
            ev.provider_record,
            record_id=ev.provider_id,
            field_pointer="/",
        )
        provenance["acquisition"] = {"retrieved_at": ev.retrieved_at}
        if ev.source == "Ticketmaster Discovery":
            provenance.update({
                "provider": "Ticketmaster Discovery API",
                "publishing_source": ev.publishing_source or None,
                "destination_host": (urlsplit(ev.url).hostname or "").removeprefix("www."),
            })
        cite = ctx.citations.register(
            ev.url or PARKS_SOURCE_URL,
            snippet="; ".join(bit for bit in snippet_bits if bit).strip(),
            title=ev.name or "NYC event", kind="DATA",
            provenance=provenance,
        )
        blocks.append(_event_block(ev, cite, now))

    window = f" for {window_start} through {window_end}" if window_end else ""
    free_scope = " whose official source evidence says free" if "free" in ctx.query.lower() else ""
    header = (
        f"{setting_note}{coverage_note}"
        f"Upcoming NYC events{window}{free_scope} from live sources (Ticketmaster + NYC Parks + "
        "NYC Permitted Events, the Street Activity Permit Office feed of street fairs, farmers "
        "markets, block parties, parades, and plaza events):\n"
    )
    catalog = header + "\n".join(blocks) if blocks else no_results
    followup = "\nOffer to narrow or list more matching events." if not resident_requested_count else ""
    return catalog + web_appendix + followup


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_nyc_events",
            description=(
                "Find structured NYC event listings by date, borough, audience, topic, or category. "
                "It combines live Ticketmaster, NYC Parks, and NYC Permitted Events into grounded, "
                "dated, linked listings. For broad requests such as things to do today, it also "
                "searches the current web in parallel so the model does not need to coordinate a "
                "second required call. Use it when the "
                "resident wants event choices or wants to filter earlier choices. Do not use it for "
                "general facts merely because they concern sports, music, or entertainment. "
                "For a specific event topic, web_search remains available for fresh context, "
                "named-event details, and gaps the structured listings do not cover. Pass `keyword` "
                "for a specific event topic, plus "
                "optional `classification`, `borough`, source-backed `audience`, and date window. "
                "For a constrained request, pass `web_query` as a short noun phrase that preserves "
                "every requested constraint so the coordinator runs one matching web lane."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Specific event topic, e.g. 'world cup' or 'jazz'. Omit for broad "
                            "requests such as 'things to do', 'events', or 'free events'."
                        ),
                    },
                    "classification": {"type": "string", "description": "Optional Ticketmaster segment: Music, Sports, Arts & Theatre, etc."},
                    "borough": {
                        "type": "string",
                        "description": (
                            "Optional NYC borough filter, e.g. 'Brooklyn'. Pass it only when "
                            "the resident names one; NYC means citywide, so omit this field."
                        ),
                    },
                    "audience": {"type": "string", "enum": ["kids"], "description": "Optional evidence-backed audience filter. Use `kids` only when the resident asks for children's events; returned rows must carry a source audience label for kids."},
                    "web_query": {
                        "type": "string",
                        "description": (
                            "Short noun phrase for the coordinator's single current-web lane. "
                            "Preserve every requested constraint, including place, date, audience, "
                            "cost, and indoor or outdoor setting. Omit for an unconstrained broad "
                            "request, which uses the resident's message."
                        ),
                    },
                    "setting": {
                        "type": "string",
                        "enum": ["indoor", "outdoor"],
                        "description": (
                            "Requested physical setting. Pass only when the resident explicitly "
                            "asks for indoor or outdoor options. Structured catalogs do not carry "
                            "this field; it preserves the constraint for the web lane and makes "
                            "the catalog limitation explicit."
                        ),
                    },
                    "window_start": {"type": "string", "format": "date", "description": "Optional ISO date (YYYY-MM-DD) the resident's timeframe starts. Pass when they name a date, range, month, or ask about past events; omit for today."},
                    "window_end": {"type": "string", "format": "date", "description": "Optional ISO date the timeframe ends; omit for open-ended."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": (
                            "Maximum structured listings to return. Default to five; pass another "
                            "count only when the resident asks for it."
                        ),
                    },
                },
            },
            handler=_handler,
            open_world=True,  # hits live Ticketmaster + Socrata
            title="Find NYC events",
        )
    ]
