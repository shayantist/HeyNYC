"""events module tool: `find_nyc_events`, a thin composition (§10.3) of shared infra.

Merges the Ticketmaster Discovery client (structured backbone) + NYC Parks public
events (query_dataset over w3wp-dpdi) into one Event shape, filtered to future dates.
No hallucinated events: every row is grounded in a live source and cited with a link.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heynyc.core.citations import data_provenance
from heynyc.core.index.corpus import clean_html
from heynyc.core.temporal import EventStatus, event_status, nyc_datetime
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
from heynyc.core.tools.geo import nyc_neighborhood_borough
from heynyc.core.tools.web_search import _TIER_LABELS

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
_PAGE_FETCH_TIMEOUT_S = 20.0
_PARK_BOROUGHS = {
    "B": "Brooklyn",
    "M": "Manhattan",
    "Q": "Queens",
    "R": "Staten Island",
    "X": "Bronx",
}
_NYC_LOCALITIES = {
    "bronx", "brooklyn", "new york", "new york city", "queens", "staten island",
}


def _nyc_locality_borough(value: object) -> str | None:
    locality = str(value or "").strip()
    if not locality:
        return None
    if locality.casefold() in _NYC_LOCALITIES:
        return locality
    return nyc_neighborhood_borough(locality)


class EventQuery(BaseModel):
    """Validated resident constraints for one event lookup."""

    model_config = ConfigDict(extra="forbid")

    keyword: str | None = Field(
        default=None,
        description=(
            "Specific named event, artist, genre, or topic, such as 'Joe Hisaishi' or 'jazz'. "
            "Do not repeat a broad classification such as Music or Sports here."
        ),
    )
    classification: str | None = Field(
        default=None,
        description="Optional Ticketmaster segment, such as Music, Sports, or Arts & Theatre.",
    )
    borough: str | None = Field(
        default=None,
        description=(
            "Optional NYC borough. Pass only when the resident names one; NYC is citywide."
        ),
    )
    audience: Literal["kids"] | None = Field(
        default=None,
        description="Use kids only when the resident asks for children's events.",
    )
    web_query: str | None = Field(
        default=None,
        description=(
            "Short noun phrase for current-web discovery preserving every requested constraint, "
            "including date, place, audience, cost, topic, and indoor or outdoor setting."
        ),
    )
    setting: Literal["indoor", "outdoor"] | None = Field(
        default=None,
        description="Optional requested physical setting: indoor or outdoor.",
    )
    has_started: bool | None = Field(
        default=None,
        description=(
            "Filter against the computed New York event interval: true for events that started, "
            "false for events that have not started, or omit when start state is unrestricted."
        ),
    )
    has_ended: bool | None = Field(
        default=None,
        description=(
            "Filter against the computed New York event interval: true for ended events, false "
            "for events not known to have ended, or omit to use the date-window default."
        ),
    )
    window_start: date | None = Field(
        default=None,
        description=(
            "ISO start date inferred from the resident's request using the current New York "
            "date. Ask a clarification instead of guessing an ambiguous date."
        ),
    )
    window_end: date | None = Field(
        default=None,
        description=(
            "ISO end date for an explicit multi-day range. Omit for a single absolute date; "
            "window_start then applies to that day only."
        ),
    )
    window_start_time: time | None = Field(
        default=None,
        description="Optional local New York lower time bound inferred from the request.",
    )
    window_end_time: time | None = Field(
        default=None,
        description="Optional local New York upper time bound inferred from the request.",
    )
    cost: Literal["free"] | None = Field(
        default=None,
        description="Use free only when the resident explicitly asks for free events.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Maximum choices requested by the resident. Set this only when the resident explicitly "
            "asks for a number; otherwise omit it and the server returns the default shortlist."
        ),
    )
    @field_validator("audience", mode="before")
    @classmethod
    def supported_audience(cls, value: str | None) -> str | None:
        if value not in {None, "kids"}:
            raise ValueError(f"Unsupported audience: {value}")
        return value

    @field_validator("setting", mode="before")
    @classmethod
    def supported_setting(cls, value: str | None) -> str | None:
        if value not in {None, "indoor", "outdoor"}:
            raise ValueError(f"Unsupported setting: {value}")
        return value


    @model_validator(mode="after")
    def ordered_window(self) -> "EventQuery":
        if self.window_start and self.window_end and self.window_end < self.window_start:
            raise ValueError("window_end must not be before window_start")
        if (
            self.window_start_time and self.window_end_time
            and (self.window_end or self.window_start) == self.window_start
            and self.window_end_time < self.window_start_time
        ):
            raise ValueError("window_end_time must not be before window_start_time")
        return self


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
    end_date: str = ""
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
    registration_info: str = ""
    evidence_excerpt: str = ""
    citation_id: str = ""
    retrieval_rank: int | None = None
    structured_source: bool = False
    category: str = ""


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
    end = dates.get("end") or {}
    start_date = _iso_date(start.get("localDate"))
    if not start_date:
        return None  # undated / TBA, don't surface as a real listing
    venues = (raw.get("_embedded") or {}).get("venues") or []
    venue = venues[0].get("name", "") if venues else ""
    borough = (venues[0].get("city") or {}).get("name", "") if venues else ""
    if borough and borough.strip().casefold() not in _NYC_LOCALITIES:
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
        end_date=_iso_date(end.get("localDate")),
        end_time=end.get("localTime") or "",
        accessibility_info=(raw.get("accessibility") or {}).get("info") or "",
        venue_accessibility=(venues[0].get("accessibleSeatingDetail") or "") if venues else "",
        publishing_source=source.get("name") or source.get("id") or "",
        provider_id=str(raw.get("id") or raw.get("url") or ""),
        provider_record=raw,
        retrieved_at=retrieved_at,
        category=" ".join(
            str((classification.get("segment") or {}).get("name") or "")
            for classification in raw.get("classifications") or ()
        ).strip(),
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
        end_date=_iso_date(raw.get("enddate")),
        end_time=_parks_time(raw.get("endtime")),
        registration_info=str(raw.get("registration_description") or "").strip(),
        provider_id=str(raw.get(":id") or url or title),
        provider_record=raw,
        retrieved_at=retrieved_at,
        category=categories,
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
    end_date = _iso_date(raw_end)
    end_time = raw_end[11:16] if len(raw_end) >= 16 else ""
    row_id = str(raw.get(":id") or "")
    return Event(
        name=name, start_date=start_date, start_time=start_time,
        venue=raw.get("event_location") or "", borough=raw.get("event_borough") or "",
        url=row_url(PERMITTED_DATASET_ID, row_id) if row_id else PERMITTED_SOURCE_URL,
        source="NYC Permitted Events", tier="authoritative",
        end_date=end_date, end_time=end_time,
        provider_id=row_id or str(raw.get("event_id") or ""),
        provider_record=raw,
        retrieved_at=retrieved_at,
    )


def _from_web_citation(citation_id: str, citation: dict, *, rank: int) -> Event:
    """Normalize one already-registered web discovery into the shared Event shape."""
    provenance = citation.get("provenance") or {}
    event_fields = provenance.get("event") or {}
    url = str(event_fields.get("url") or citation.get("url") or "")
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    evidence_excerpt = str(citation.get("snippet") or "")
    if event_fields:
        evidence_excerpt = "; ".join(
            str(value)
            for value in (
                event_fields.get("name"), event_fields.get("start_date"),
                event_fields.get("start_time"), event_fields.get("venue"),
                event_fields.get("category"),
            )
            if value
        )
    return Event(
        name=str(event_fields.get("name") or citation.get("title") or host or "Web event lead"),
        start_date=_iso_date(event_fields.get("start_date")),
        start_time=str(event_fields.get("start_time") or ""),
        end_date=_iso_date(event_fields.get("end_date")),
        end_time=str(event_fields.get("end_time") or ""),
        venue=str(event_fields.get("venue") or ""),
        borough=_nyc_locality_borough(event_fields.get("borough")) or "",
        url=url,
        source="Web discovery",
        tier=str(provenance.get("source_tier") or "unverified"),
        publishing_source=host,
        provider_id=url,
        provider_record=event_fields or citation,
        retrieved_at=str((provenance.get("acquisition") or {}).get("fetched_at") or ""),
        evidence_excerpt=evidence_excerpt,
        citation_id=citation_id,
        retrieval_rank=rank,
        structured_source=bool(event_fields),
        category=str(event_fields.get("category") or ""),
    )


def _from_web_citation_events(citation_id: str, citation: dict, *, rank: int) -> list[Event]:
    structured = (citation.get("provenance") or {}).get("events") or []
    if not structured:
        return [_from_web_citation(citation_id, citation, rank=rank)]
    return [
        _from_web_citation(
            citation_id,
            {
                **citation,
                "provenance": {**(citation.get("provenance") or {}), "event": event},
            },
            rank=rank + offset,
        )
        for offset, event in enumerate(structured)
        if _nyc_locality_borough(event.get("borough")) is not None
    ]


def _future_only(events: list[Event], today: str) -> list[Event]:
    """Keep events whose source interval overlaps or follows `today`."""
    return [e for e in events if max(e.start_date, e.end_date or "") >= today]


def _not_ended_today(events: list[Event], now: datetime) -> list[Event]:
    """Legacy strict filter retained for callers that require confirmed attendability."""
    return [
        event
        for event in events
        if event.start_date != now.date().isoformat()
        or _event_temporal_status(event, now) in {"upcoming", "in_progress"}
    ]


def _shortlist(events: list[Event], limit: int) -> list[Event]:
    """Cap structured rows while preserving source and requested-date coverage."""
    unique: list[Event] = []
    identities: set[tuple[str, ...]] = set()
    urls: set[str] = set()
    generic_urls = {
        DISCOVERY_URL.casefold(), PARKS_SOURCE_URL.casefold(), PERMITTED_SOURCE_URL.casefold(),
    }
    for event in events:
        url = event.url.strip().casefold()
        if url and url not in generic_urls and url in urls:
            continue
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
        if url and url not in generic_urls:
            urls.add(url)
        identities.add(identity)
        unique.append(event)
    seen: dict[tuple[str, str], int] = {}
    ranked: list[tuple[int, str, Event]] = []
    for event in sorted(unique, key=lambda e: e.start_date or "9999-12-31"):
        group = (event.source, event.start_date)
        rank = event.retrieval_rank if event.retrieval_rank is not None else seen.get(group, 0)
        seen[group] = rank + 1
        ranked.append((rank, event.start_date, event))
    ranked.sort(key=lambda item: (
        item[0],
        item[1] or "9999-12-31",
        _parse_start_time(item[2].start_time) or datetime.max.time(),
    ))
    return [event for _, _, event in ranked[:limit]]


_NO_RESULTS = (
    "No NYC events matched that from the live sources (Ticketmaster + NYC Parks + "
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


def _event_temporal_status(ev: Event, now: datetime) -> EventStatus:
    """Normalize one event's raw local fields and classify it against the NYC clock."""
    if not ev.start_date:
        return "unknown"
    event_day = date.fromisoformat(ev.start_date)
    start_time = _parse_start_time(ev.start_time)
    if start_time is None:
        if event_day < now.date():
            return "ended"
        return "upcoming" if event_day > now.date() else "unknown"
    start_at = nyc_datetime(event_day, start_time)
    end_time = _parse_start_time(ev.end_time)
    end_at = None
    if end_time is not None:
        end_day = date.fromisoformat(ev.end_date) if ev.end_date else event_day
        if not ev.end_date and end_time <= start_time:
            end_day += timedelta(days=1)
        end_at = nyc_datetime(end_day, end_time)
    return event_status(start_at, end_at, now)


def _temporal_filter(
    events: list[Event], *, has_started: bool | None, has_ended: bool | None, now: datetime,
) -> list[Event]:
    kept = []
    for event in events:
        status = _event_temporal_status(event, now)
        if status == "unknown":
            if has_ended is not True:
                kept.append(event)
            continue
        started = status in {"in_progress", "ended"}
        ended = status == "ended"
        if has_started is not None and started != has_started:
            continue
        if has_ended is not None and ended != has_ended:
            continue
        kept.append(event)
    return kept


def _temporal_instruction(has_started: bool | None, has_ended: bool | None) -> str:
    if has_ended is True:
        return "include only ended events and describe them in the past tense"
    if has_started is True:
        return (
            "recommend only events computed in progress; keep unknown-time leads separate and "
            "say they are not confirmed happening now"
        )
    if has_started is False:
        return "include only events that have not started; keep unknown-time leads separate"
    return "exclude events known to have ended"


def _today_timing_note(ev: Event, now: datetime) -> str:
    """Expose the deterministic status while retaining the source's raw times."""
    status = _event_temporal_status(ev, now)
    if not ev.start_date:
        return "; date and timing not confirmed"
    if ev.start_date != now.date().isoformat():
        return "; ended" if status == "ended" else ""
    start = _parse_start_time(ev.start_time)
    if status == "unknown" and start is None:
        return "; timing unknown, not confirmed currently attendable"
    if status == "unknown":
        return "; already started or ended earlier today; end time unknown"
    return {
        "upcoming": "; starts later today",
        "in_progress": "; in progress",
        "ended": "; ended",
        "unknown": "",
    }[status]


def _event_block(ev: Event, cite: str, now: Optional[datetime] = None) -> str:
    if ev.start_date:
        weekday = date.fromisoformat(ev.start_date).strftime("%A")
        when = f"{weekday}, {ev.start_date}" + (f" {ev.start_time}" if ev.start_time else "")
    else:
        when = "date and time not normalized from this source"
    where = f" @ {ev.venue}" if ev.venue else ""
    timing = _today_timing_note(ev, now) if now else ""
    end_when = ev.end_time
    if ev.end_date and ev.end_date != ev.start_date:
        end_when = f"{date.fromisoformat(ev.end_date).strftime('%A')}, {ev.end_date} {ev.end_time}"
    end = f"; ends {end_when}" if end_when else ""
    free = "; free" if ev.free_evidence else ""
    audience = f"; {ev.audience}" if ev.audience else ""
    registration = f"; registration: {ev.registration_info}" if ev.registration_info else ""
    status = f"; status {ev.provider_status}" if ev.provider_status else ""
    accessibility = (
        f"; accessibility: {ev.accessibility_info or ev.venue_accessibility}"
        if ev.accessibility_info or ev.venue_accessibility
        else ""
    )
    details = f"\n  Details: {ev.url}" if ev.url else ""
    evidence = f"\n  Source excerpt: {ev.evidence_excerpt}" if ev.evidence_excerpt else ""
    source = ev.source
    match_warning = ""
    if ev.source == "Web discovery":
        source = f"{source}; {_TIER_LABELS.get(ev.tier, ev.tier)}"
        if not ev.structured_source:
            match_warning = "; unconfirmed lead, not a matching option until its constraints are checked"
    return (
        f"- {ev.name}{where}, {when}{timing}{end}{free}{audience}{registration}{status}"
        f"{match_warning}"
        f"{accessibility} "
        f"({source}) {{cite:{cite}}}{evidence}{details}"
    )


def _explicitly_free(events: list[Event], cost: Literal["free"] | None) -> list[Event]:
    if cost != "free":
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
    blob = " ".join(
        (event.name, event.venue, event.audience, event.free_evidence, event.evidence_excerpt)
    ).lower()
    return not terms or any(term in blob for term in terms)


def _matches_classification(event: Event, classification: str) -> bool:
    terms = _keyword_terms(classification)
    blob = " ".join(
        (event.name, event.category or event.evidence_excerpt)
    ).casefold()
    return not terms or any(term in blob for term in terms)


def _event_discovery_domains(ctx: ToolContext) -> list[str]:
    return sorted({
        domain
        for module in ctx.registry.modules
        if module.name == "events"
        for tier in ("editorial", "community")
        for domain in module.source_tiers.get(tier, ())
    })


def _is_generic_event_page(url: str) -> bool:
    return urlsplit(url).path.rstrip("/").casefold() in {"", "/discover", "/event", "/events"}


async def _handler(args: dict, ctx: ToolContext) -> str:
    query = EventQuery.model_validate(args)
    keyword = (query.keyword or "").strip() or None
    if keyword and not _keyword_terms(keyword):
        keyword = None
    classification = (query.classification or "").strip() or None
    borough = (query.borough or "").strip().lower()
    audience = (query.audience or "").strip().lower()
    web_query = (query.web_query or "").strip()
    setting = (query.setting or "").strip().lower()
    resident_requested_count = query.max_results is not None
    max_results = query.max_results or 5

    now = datetime.now(NYC_TZ)
    today = now.strftime("%Y-%m-%d")

    # The agent extracts the resident's date window. Python validates it and computes event state.
    arg_start = query.window_start.isoformat() if query.window_start else None
    arg_end = query.window_end.isoformat() if query.window_end else None
    if arg_start:
        window_start, window_end = arg_start, arg_end or arg_start
    elif arg_end:
        window_start, window_end = today, arg_end
    else:
        window_start = window_end = today
    has_ended = query.has_ended
    if has_ended is None:
        has_ended = bool(window_end and window_end < today)
    temporal_instruction = _temporal_instruction(query.has_started, has_ended)
    start_time_bound = query.window_start_time
    end_time_bound = query.window_end_time

    def local_bound(day: date, value: time) -> str:
        return datetime.combine(day, value, tzinfo=NYC_TZ).astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def utc_midnight(day: date) -> str:
        return datetime.combine(day, datetime.min.time(), tzinfo=NYC_TZ).astimezone(
            timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    start_day = date.fromisoformat(window_start)
    end_day = date.fromisoformat(window_end) if window_end else None
    start_dt = (
        local_bound(start_day, start_time_bound)
        if start_time_bound
        else utc_midnight(start_day)
        if window_start != today or has_ended or query.has_started is True
        else now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    end_dt = (
        local_bound(end_day, end_time_bound)
        if end_day and end_time_bound
        else utc_midnight(end_day + timedelta(days=1))
        if end_day
        else None
    )
    ticketmaster_keyword = keyword
    if keyword and classification and keyword.casefold() == classification.casefold():
        ticketmaster_keyword = None

    async def ticketmaster_source():
        return await ticketmaster_events(
            keyword=ticketmaster_keyword,
            classification=classification,
            start_datetime=start_dt,
            end_datetime=end_dt,
            size=200,
            client=ctx.http,
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
        (bool(web_query) or ctx.event_turn == "discovery" or (not keyword and not classification))
        and bool(ctx.query.strip())
        and ctx.toolbox is not None
        and "web_search" in ctx.toolbox
    )
    search_query = ""
    calendar_urls: list[str] = []
    web_citations_before = set(ctx.citations.mapping())
    web_citation_cursor = ctx.citations.touch_cursor()
    if broad_web:
        query_terms = [
            term
            for term in (keyword or classification, borough, audience, setting, query.cost)
            if term
        ]
        start_day = date.fromisoformat(window_start)
        search_parts = [
            "NYC",
            *query_terms,
            "events",
            f"{start_day.strftime('%B')} {start_day.day}, {start_day.year}",
        ]
        if window_end and window_end != window_start:
            end_day = date.fromisoformat(window_end)
            search_parts.append(
                f"to {end_day.strftime('%B')} {end_day.day}, {end_day.year}"
            )
        search_query = " ".join(search_parts) if query_terms else web_query or ctx.query.strip()
        web_args = {"query": search_query, "count": 10}
        if preferred_domains := _event_discovery_domains(ctx):
            web_args["prefer"] = preferred_domains
        sources.append((
            "broad_web",
            ctx.toolbox["web_search"].handler(web_args, ctx),
        ))
        # ponytail: direct calendars currently cover single-day lookups. Range searches keep
        # using the bounded web lane until a real recall failure justifies several page fetches.
        if window_end == window_start and "web_fetch" in ctx.toolbox:
            calendar_urls = list(dict.fromkeys(
                template.format(date=start_day)
                for module in ctx.registry.modules
                if module.name == "events"
                for template in module.event_calendar_templates
            ))
            sources.extend(
                (
                    f"event_calendar_{index}",
                    ctx.toolbox["web_fetch"].handler(
                        {"url": url, "query": search_query}, ctx,
                    ),
                )
                for index, url in enumerate(calendar_urls)
            )
    gathered = await asyncio.gather(*(
        asyncio.wait_for(
            call,
            timeout=(
                _PAGE_FETCH_TIMEOUT_S
                if name.startswith("event_calendar_")
                else _SOURCE_TIMEOUT_S
            ),
        )
        for name, call in sources
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
    web_results = [
        result for name, result in results.items()
        if name == "broad_web" or name.startswith("event_calendar_")
    ]
    web_failed = bool(web_results) and all(
        isinstance(result, BaseException) for result in web_results
    )
    web_context = "\n\n".join(
        str(result) for result in web_results
        if result and not isinstance(result, BaseException)
    ) or None
    added = ctx.citations.mapping()
    new_citations = [
        citation
        for citation_id, citation in added.items()
        if citation_id not in web_citations_before
    ]
    generic = next(
        (
            citation
            for citation in new_citations[:1]
            if _is_generic_event_page(citation["url"])
        ),
        None,
    )
    resolved_primary = next(
        (
            citation
            for citation in new_citations
            if not _is_generic_event_page(citation["url"])
            and (
                generic is None
                or (urlsplit(citation["url"]).hostname or "").removeprefix("www.")
                == (urlsplit(generic["url"]).hostname or "").removeprefix("www.")
            )
        ),
        None,
    )
    if generic is not None and ctx.toolbox is not None and "web_search" in ctx.toolbox:
        host = (urlsplit(generic["url"]).hostname or "").removeprefix("www.")
        resolved_primary = next(
            (
                citation
                for citation_id, citation in added.items()
                if citation_id not in web_citations_before
                and citation["url"] != generic["url"]
                and (urlsplit(citation["url"]).hostname or "").removeprefix("www.") == host
                and not _is_generic_event_page(citation["url"])
            ),
            None,
        )
        focused_query = " ".join(
            part for part in (generic.get("title", ""), generic.get("snippet", "")[:240])
            if part
        )
        if resolved_primary is None and host and focused_query:
            focused_citations_before = set(added)
            try:
                focused = await asyncio.wait_for(
                    ctx.toolbox["web_search"].handler(
                        {"query": focused_query, "prefer": [host], "count": 5}, ctx,
                    ),
                    timeout=_SOURCE_TIMEOUT_S,
                )
            except Exception:
                focused = ""
            if focused:
                web_context = f"{web_context}\n\nFocused direct-page search:\n{focused}"
                resolved_primary = next(
                    (
                        citation
                        for citation_id, citation in ctx.citations.mapping().items()
                        if citation_id not in focused_citations_before
                        and (urlsplit(citation["url"]).hostname or "").removeprefix("www.")
                        == host
                        and not _is_generic_event_page(citation["url"])
                    ),
                    None,
                )
    if (
        web_context
        and ctx.toolbox is not None
        and "web_fetch" in ctx.toolbox
    ):
        added = ctx.citations.mapping()
        direct_candidates = [
            citation
            for citation_id, citation in added.items()
            if citation_id not in web_citations_before
            and not _is_generic_event_page(citation["url"])
        ]
        if resolved_primary is not None:
            direct_candidates.insert(0, resolved_primary)
        requested_dates = {
            value
            for day in (start_day, end_day)
            if day is not None
            for value in (
                day.isoformat(),
                f"{day.strftime('%B')} {day.day}",
            )
        }
        requested_topics = _keyword_terms(keyword or classification or "")

        def fetch_priority(citation: dict) -> tuple[bool, bool]:
            blob = " ".join(
                str(citation.get(field) or "") for field in ("title", "snippet", "url")
            ).casefold()
            return (
                any(value.casefold() in blob for value in requested_dates),
                any(topic in blob for topic in requested_topics),
            )

        if generic is not None and resolved_primary is not None:
            primaries = [resolved_primary]
        else:
            direct_candidates.sort(key=fetch_priority, reverse=True)
            primaries = list(
                {citation["url"]: citation for citation in direct_candidates}.values()
            )[:2]

        async def fetch_primary(primary: dict) -> str:
            try:
                return await asyncio.wait_for(
                    ctx.toolbox["web_fetch"].handler(
                        {"url": primary["url"], "query": search_query},
                        ctx,
                    ),
                    timeout=_PAGE_FETCH_TIMEOUT_S,
                )
            except Exception:
                return ""

        fetched_pages = await asyncio.gather(*(fetch_primary(primary) for primary in primaries))
        for fetched in fetched_pages:
            if fetched:
                web_context = f"{web_context}\n\nSelected event page:\n{fetched}"

        current_fetch_citation_ids = ctx.citations.touched_since(web_citation_cursor)
        attempted_urls = {
            citation["url"]
            for citation_id, citation in ctx.citations.mapping().items()
            if citation_id in current_fetch_citation_ids
            if (citation.get("provenance") or {}).get("acquisition")
            or (citation.get("provenance") or {}).get("evidence_grade") == "unavailable"
        }
        child_pages: list[dict[str, str]] = []
        for citation_id, citation in ctx.citations.mapping().items():
            if citation_id not in current_fetch_citation_ids:
                continue
            for event_fields in (citation.get("provenance") or {}).get("events") or ():
                event_url = str(event_fields.get("url") or "")
                if not event_url or event_url in attempted_urls or event_fields.get("borough"):
                    continue
                event_end = str(event_fields.get("end_date") or event_fields.get("start_date") or "")
                event_start = str(event_fields.get("start_date") or "")
                if not event_start or event_end < window_start or (
                    window_end and event_start > window_end
                ):
                    continue
                candidate = _from_web_citation(
                    citation_id,
                    {
                        **citation,
                        "provenance": {
                            **(citation.get("provenance") or {}), "event": event_fields,
                        },
                    },
                    rank=0,
                )
                if (keyword or classification) and not _matches_keyword(
                    candidate, keyword or classification or "",
                ):
                    continue
                attempted_urls.add(event_url)
                child_pages.append({"url": event_url})
                if len(child_pages) == 2:
                    break
            if len(child_pages) == 2:
                break
        fetched_children = await asyncio.gather(
            *(fetch_primary(primary) for primary in child_pages)
        )
        for fetched in fetched_children:
            if fetched:
                web_context = f"{web_context}\n\nVerified direct event page:\n{fetched}"
    current_web_citation_ids = ctx.citations.touched_since(web_citation_cursor)
    raw_web_citation_items = [
        (citation_id, citation)
        for citation_id, citation in ctx.citations.mapping().items()
        if citation_id in current_web_citation_ids
    ]
    web_citation_ids = {citation_id for citation_id, _citation in raw_web_citation_items}
    citation_by_url: dict[str, tuple[str, dict]] = {}
    url_order: list[str] = []
    evidence_priority = {
        "unavailable": 0, "discovery": 1, "search_excerpt": 2,
        "authoritative_excerpt": 3, "fetched": 4, "authoritative": 5,
    }
    for citation_id, citation in raw_web_citation_items:
        url = citation["url"]
        existing = citation_by_url.get(url)
        if existing is None:
            url_order.append(url)
        if existing is None or evidence_priority.get(
            (citation.get("provenance") or {}).get("evidence_grade", ""), 0,
        ) > evidence_priority.get(
            (existing[1].get("provenance") or {}).get("evidence_grade", ""), 0,
        ):
            citation_by_url[url] = (citation_id, citation)
    web_citation_items = [citation_by_url[url] for url in url_order]
    direct_web_items = [
        item for item in web_citation_items if not _is_generic_event_page(item[1]["url"])
    ]
    if direct_web_items:
        web_citation_items = direct_web_items
    calendar_event_ranks = {
        str(event.get("url") or ""): rank
        for _citation_id, citation in web_citation_items
        if citation["url"] in calendar_urls
        for rank, event in enumerate((citation.get("provenance") or {}).get("events") or ())
        if event.get("url")
    }
    web_limitation = (
        "Current web event leads were unavailable. Results are partial."
        if web_failed
        else ""
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
    events += [
        event
        for rank, (citation_id, citation) in enumerate(web_citation_items)
        for event in _from_web_citation_events(
            citation_id,
            citation,
            rank=(
                0
                if citation["url"] in calendar_urls
                else calendar_event_ranks.get(citation["url"], rank)
            ),
        )
    ]
    if keyword:
        events = [
            event for event in events
            if event.source == "Ticketmaster Discovery"
            or _matches_keyword(event, keyword)
        ]
    elif classification:
        events = [
            event for event in events
            if event.source == "Ticketmaster Discovery"
            or _matches_classification(event, classification)
        ]

    def _window_filter(rows: list[Event]) -> list[Event]:
        incomplete = [event for event in rows if not event.start_date]
        kept = _future_only([event for event in rows if event.start_date], window_start)
        if window_end:
            kept = [e for e in kept if e.start_date <= window_end]
        if ctx.event_turn != "preparation" or query.has_started is not None or query.has_ended is not None:
            kept = _temporal_filter(
                kept, has_started=query.has_started, has_ended=has_ended, now=now,
            )
        kept = _explicitly_free(kept, query.cost)
        if start_time_bound or end_time_bound:
            kept = [
                event for event in kept
                if (parsed := _parse_start_time(event.start_time)) is not None
                and (start_time_bound is None or parsed >= start_time_bound)
                and (end_time_bound is None or parsed <= end_time_bound)
            ]
        if borough:
            kept = [e for e in kept if borough in e.borough.lower()]
        if audience == "kids":
            kept = [e for e in kept if "kids" in e.audience.lower()]
        kept += incomplete
        return _shortlist(kept, max_results)

    events = _window_filter(events)
    selected_web_citation_ids = {event.citation_id for event in events if event.citation_id}
    ctx.citations.discard(
        (web_citation_ids - web_citations_before) - selected_web_citation_ids
    )
    ctx.event_discovery_citation_ids.update(selected_web_citation_ids)
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
        return "\n\n".join(
            part for part in (setting_note + no_results, web_limitation) if part
        )

    blocks = []
    for ev in events:
        if ev.citation_id and not ev.structured_source:
            blocks.append(_event_block(ev, ev.citation_id, now))
            continue
        weekday = date.fromisoformat(ev.start_date).strftime("%A")
        # The snapshot carries the FULL row the model will describe, time and source included,
        # so cited prose stays supported by its own evidence.
        timing = _today_timing_note(ev, now).removeprefix("; ")
        temporal_status = _event_temporal_status(ev, now)
        snippet_bits = [
            f"Name: {ev.name}",
            f"Date: {weekday}, {ev.start_date}",
            f"Start time: {ev.start_time}" if ev.start_time else "",
            f"End date: {ev.end_date}" if ev.end_date else "",
            f"End time: {ev.end_time}" if ev.end_time else "",
            f"Venue: {ev.venue}" if ev.venue else "",
            f"Borough: {ev.borough}" if ev.borough else "",
            f"Timing at lookup: {timing}" if timing else "",
            f"Temporal status: {temporal_status}",
            f"Evaluated at: {now.isoformat()}",
            f"Cost evidence: {ev.free_evidence}" if ev.free_evidence else "",
            f"Audience: {ev.audience}" if ev.audience else "",
            f"Registration: {ev.registration_info}" if ev.registration_info else "",
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
        derivation = {
            "temporal": {
                "start_date": ev.start_date,
                "start_time": ev.start_time,
                "end_date": ev.end_date,
                "end_time": ev.end_time,
                "evaluated_at": now.isoformat(),
                "status": temporal_status,
            },
        }
        if ev.structured_source:
            derivation["source_citation_id"] = ev.citation_id
        provenance = data_provenance(
            ev.provider_record,
            record_id=ev.provider_id,
            field_pointer="/",
            derivation=derivation,
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
    free_scope = " whose official source evidence says free" if query.cost == "free" else ""
    header = (
        f"{setting_note}{coverage_note}"
        f"NYC event candidates{window}{free_scope} from live catalog and web sources. "
        "A web lead with unknown fields stays explicitly incomplete until its source establishes "
        "them:\n"
    )
    catalog = header + "\n".join(blocks) if blocks else no_results
    followup = (
        "\nThis is a shortlist, not every matching event. Offer to narrow or list more."
        if not resident_requested_count
        else ""
    )
    return (
        "These candidates already passed one shared bounded shortlist. Choose by exact date and "
        "topic match first, then apply NYC location and this "
        f"temporal rule: {temporal_instruction}, before relevance and variety. "
        "Source tier describes evidence confidence, not how "
        "interesting an event is. A normalized web record already includes its deterministic "
        "status and per-event citation. Before making a time-status claim from an excerpt-only "
        "web lead, call evaluate_event_time with its exact name, date, and available times and "
        "cite its derived result. Omit unavailable values so the result stays unknown instead of "
        "guessing. Resolve a general calendar URL to the direct event page with "
        "one focused web search. Preserve a useful link as an unconfirmed lead when its excerpt "
        "does not establish every hard constraint. An exact-date editorial listing excerpt "
        "remains usable evidence for only the event names, dates, venues, and times it explicitly "
        "states when its page cannot be fetched. For general discovery, include a matching "
        "non-marketplace candidate when available instead of returning only marketplace "
        "candidates:\n"
        + "\n\n".join(part for part in (catalog, web_limitation) if part)
        + followup
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_nyc_events",
            description=(
                "Find structured NYC event listings by date, borough, audience, topic, or category. "
                "It combines live Ticketmaster, NYC Parks, and NYC Permitted Events into grounded, "
                "dated, linked listings. For event discovery requests, it also "
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
                " Pass the inferred ISO date and optional local time window. Use `has_started` "
                "and `has_ended` only for a requested interval relation; the server computes "
                "those values against the New York clock."
            ),
            parameters=EventQuery.model_json_schema(),
            handler=_handler,
            open_world=True,  # hits live Ticketmaster + Socrata
            title="Find NYC events",
        ),
    ]
