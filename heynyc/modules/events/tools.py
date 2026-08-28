"""Event normalization and dormant structured event connectors.

Production exposes only `extract_events`, which turns selected fetched evidence into
the existing `Event` shape and applies deterministic date and time constraints.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from heynyc.core.citations import data_provenance
from heynyc.core.grounding import check_grounding, citation_evidence
from heynyc.core.index.corpus import clean_html
from heynyc.core.location import LocationRequest
from heynyc.core.temporal import EventStatus, interval_status, nyc_datetime
from heynyc.core.ticketmaster import (
    DISCOVERY_URL,
    TicketmasterSearchResult,
    ticketmaster_events,
)
from heynyc.core.tools.base import Tool, ToolContext, ToolInput
from heynyc.core.tools.datasets import (
    dataset_url,
    query_dataset,
    query_dataset_pages,
    row_url,
)
from heynyc.core.tools.geo import (
    current_resolved_location,
    miles,
    nyc_neighborhood_borough,
    rank_nearby,
    resident_supplied_location,
    resolve_location,
)
from heynyc.core.tools.web_fetch import _url_safe_shape
from heynyc.core.tools.web_search import (
    _TIER_LABELS,
    _text_tokens,
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
_PARK_BOROUGHS = {
    "B": "Brooklyn",
    "M": "Manhattan",
    "Q": "Queens",
    "R": "Staten Island",
    "X": "Bronx",
}
_NYC_LOCALITIES = {
    "bronx", "brooklyn", "manhattan", "new york", "new york city", "queens",
    "staten island",
}


def _nyc_locality_borough(value: object) -> str | None:
    locality = str(value or "").strip()
    if not locality:
        return None
    if locality.casefold() in _NYC_LOCALITIES:
        return locality
    return nyc_neighborhood_borough(locality)


class EventQuery(LocationRequest):
    """Resident-visible constraints for one event lookup."""

    starts_after: AwareDatetime | None = Field(
        default=None,
        description="Inclusive lower bound for event start time, with a UTC offset.",
    )
    starts_before: AwareDatetime | None = Field(
        default=None,
        description="Exclusive upper bound for event start time, with a UTC offset.",
    )
    active_at: AwareDatetime | None = Field(
        default=None,
        description="Time when the event must be in progress, with a UTC offset.",
    )
    topic: str | None = Field(
        default=None,
        description="Requested artist, event name, genre, activity, or cultural theme.",
    )
    borough: str | None = Field(
        default=None,
        description="NYC borough, only when the resident names one; NYC is citywide.",
    )
    audience: Literal["kids"] | None = Field(
        default=None,
        description="Use kids only for a children's event request.",
    )
    setting: Literal["indoor", "outdoor"] | None = Field(
        default=None,
        description="Requested indoor or outdoor setting.",
    )
    cost: Literal["free"] | None = Field(
        default=None,
        description="Use free only when the resident explicitly asks for free events.",
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
        if self.starts_after and self.starts_before and self.starts_before <= self.starts_after:
            raise ValueError("starts_before must be after starts_after")
        if self.active_at and (self.starts_after or self.starts_before):
            raise ValueError("active_at cannot be combined with a start-time range")
        return self


@dataclass
class Event:
    name: str
    start_date: str  # YYYY-MM-DD
    start_time: str = ""  # local time string, possibly ""
    venue: str = ""
    borough: str = ""
    url: SkipJsonSchema[str] = ""
    source: SkipJsonSchema[str] = ""
    tier: SkipJsonSchema[str] = ""
    free_evidence: str = ""
    audience: str = ""
    end_date: str = ""
    end_time: str = ""
    timezone: SkipJsonSchema[str] = ""
    accessibility_info: str = ""
    registration_info: str = ""
    citation_id: str = ""
    evidence_excerpt: str = ""
    category: SkipJsonSchema[str] = ""
    lat: SkipJsonSchema[float | None] = None
    lon: SkipJsonSchema[float | None] = None
    provider_status: SkipJsonSchema[str] = ""
    public_sale_start: SkipJsonSchema[str] = ""
    public_sale_end: SkipJsonSchema[str] = ""
    public_sale_start_tbd: SkipJsonSchema[bool | None] = None
    venue_accessibility: SkipJsonSchema[str] = ""
    publishing_source: SkipJsonSchema[str] = ""
    provider_id: SkipJsonSchema[str] = ""
    provider_record: SkipJsonSchema[dict] = field(default_factory=dict)
    retrieved_at: SkipJsonSchema[str] = ""
    retrieval_rank: SkipJsonSchema[int | None] = None
    structured_source: SkipJsonSchema[bool] = False
    distance_miles: SkipJsonSchema[float | None] = None


class ExtractEventsInput(ToolInput):
    starts_after: AwareDatetime | None = Field(None, description="Earliest start")
    starts_before: AwareDatetime | None = Field(None, description="Latest start")
    active_at: AwareDatetime | None = Field(None, description="Must be underway then")
    max_results: int | None = Field(None, ge=1, description="Resident-requested count")
    audience: Literal["kids"] | None = Field(None, description="Kids only")
    cost: Literal["free"] | None = Field(None, description="Free only")
    events: list[Event] = Field(
        default_factory=list,
        description="Events extracted from fetched page text; cite each source",
    )

    @model_validator(mode="after")
    def ordered_window(self) -> "ExtractEventsInput":
        if self.starts_after and self.starts_before and self.starts_before <= self.starts_after:
            raise ValueError("starts_before must be after starts_after")
        if self.active_at and (self.starts_after or self.starts_before):
            raise ValueError("active_at cannot be combined with a start-time range")
        return self


def _coordinates(latitude: object, longitude: object) -> tuple[float | None, float | None]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None, None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else (None, None)


def _coordinate_text(value: object) -> tuple[float | None, float | None]:
    parts = str(value or "").split(",", 1)
    return _coordinates(*parts) if len(parts) == 2 else (None, None)


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
    latitude, longitude = _coordinates(
        (venues[0].get("location") or {}).get("latitude") if venues else None,
        (venues[0].get("location") or {}).get("longitude") if venues else None,
    )
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
        audience=(raw.get("ageRestrictions") or {}).get("ageRuleDescription") or "",
        publishing_source=source.get("name") or source.get("id") or "",
        provider_id=str(raw.get("id") or raw.get("url") or ""),
        provider_record=raw,
        retrieved_at=retrieved_at,
        category=" ".join(
            str((classification.get("segment") or {}).get("name") or "")
            for classification in raw.get("classifications") or ()
        ).strip(),
        lat=latitude,
        lon=longitude,
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
    registration_link = raw.get("registration_url")
    registration_url = (
        registration_link.get("url", "")
        if isinstance(registration_link, dict)
        else (registration_link or "")
    )
    if _url_safe_shape(str(registration_url)):
        url = str(registration_url)
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
    latitude, longitude = _coordinate_text(raw.get("coordinates"))
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
        lat=latitude,
        lon=longitude,
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
    source_url = str(citation.get("url") or "")
    candidate_url = str(event_fields.get("url") or source_url)
    url = candidate_url if _url_safe_shape(candidate_url) else source_url
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
        free_evidence=str(event_fields.get("free_evidence") or ""),
        audience=str(event_fields.get("audience") or ""),
        publishing_source=host,
        provider_id=url,
        provider_record=(
            {
                "event": event_fields,
                "source_url": str(citation.get("url") or ""),
                "source_provenance": provenance,
            }
            if event_fields
            else citation
        ),
        retrieved_at=str((provenance.get("acquisition") or {}).get("fetched_at") or ""),
        evidence_excerpt=evidence_excerpt,
        citation_id=citation_id,
        retrieval_rank=rank,
        structured_source=bool(event_fields),
        category=str(event_fields.get("category") or ""),
    )


def _from_web_citation_events(citation_id: str, citation: dict, *, rank: int) -> list[Event]:
    provenance = citation.get("provenance") or {}
    structured = provenance.get("events") or []
    if event := provenance.get("event"):
        structured = [event]
    if not structured:
        return []
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


def _schema_nodes(value: object):
    if isinstance(value, list):
        for item in value:
            yield from _schema_nodes(item)
    elif isinstance(value, dict):
        yield value
        yield from _schema_nodes(value.get("@graph"))


def _schema_date_time(value: object) -> tuple[str, str]:
    raw = str(value or "").strip()
    if len(raw) == 10:
        return _iso_date(raw), ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "", ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(NYC_TZ)
    return parsed.date().isoformat(), parsed.strftime("%H:%M:%S")


def _schema_event(value: dict, citation_id: str, citation: dict) -> Event | None:
    schema_type = value.get("@type")
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    if not any(str(item or "").casefold().endswith("event") for item in types):
        return None
    start = str(value.get("startDate") or "")
    start_date, start_time = _schema_date_time(start)
    if not value.get("name") or not start_date:
        return None
    end = str(value.get("endDate") or "")
    end_date, end_time = _schema_date_time(end)
    location = value.get("location") if isinstance(value.get("location"), dict) else {}
    address = location.get("address")
    address = address if isinstance(address, dict) else {}
    locality = str(address.get("addressLocality") or "")
    borough = _nyc_locality_borough(locality) or ""
    offers = value.get("offers")
    offers = offers if isinstance(offers, list) else [offers]
    free = any(
        isinstance(offer, dict) and str(offer.get("price") or "").strip() in {"0", "0.0", "0.00"}
        for offer in offers
    )
    source_url = str(citation.get("url") or "")
    event_url = str(value.get("url") or source_url)
    return Event(
        name=str(value["name"]),
        start_date=start_date,
        start_time=start_time,
        end_date=end_date,
        end_time=end_time,
        venue=str(location.get("name") or ""),
        borough=borough,
        url=event_url if _url_safe_shape(event_url) else source_url,
        free_evidence="Schema.org offer price is 0" if free else "",
        audience=str(value.get("typicalAgeRange") or ""),
        accessibility_info=str(value.get("accessibilityFeature") or ""),
        citation_id=citation_id,
        category=str(value.get("eventType") or ""),
        source="Web discovery",
        tier=str((citation.get("provenance") or {}).get("source_tier") or "unverified"),
        provider_id=event_url,
        provider_record=value,
        evidence_excerpt=str(citation.get("snippet") or ""),
        structured_source=True,
    )


def _future_only(events: list[Event], today: str) -> list[Event]:
    """Keep events whose source interval overlaps or follows `today`."""
    return [e for e in events if max(e.start_date, e.end_date or "") >= today]


def _dedupe_order(
    events: list[Event],
    limit: int | None,
) -> list[Event]:
    """Deduplicate records, order them deterministically, then apply an explicit count."""
    unique: list[Event] = []
    identities: set[tuple[str, ...]] = set()
    url_names: set[tuple[str, str]] = set()
    generic_urls = {
        DISCOVERY_URL.casefold(), PARKS_SOURCE_URL.casefold(), PERMITTED_SOURCE_URL.casefold(),
    }
    for event in events:
        url = event.url.strip().casefold()
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
        url_name = (url, event.name.casefold())
        if url and url not in generic_urls and url_name in url_names:
            continue
        if identity in identities:
            continue
        if url and url not in generic_urls:
            url_names.add(url_name)
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
    ordered = [event for _, _, event in ranked]
    return ordered[:limit] if limit is not None else ordered


_NO_RESULTS = (
    "No matching structured event records were returned."
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
    end_day = date.fromisoformat(ev.end_date) if ev.end_date else None
    if event_day < now.date() and (end_day is None or end_day < now.date()):
        return "ended"
    start_time = _parse_start_time(ev.start_time)
    if start_time is None:
        return "upcoming" if event_day > now.date() else "unknown"
    start_at = nyc_datetime(event_day, start_time)
    end_time = _parse_start_time(ev.end_time)
    end_at = None
    if end_time is not None:
        end_day = end_day or event_day
        if not ev.end_date and end_time <= start_time:
            end_day += timedelta(days=1)
        end_at = nyc_datetime(end_day, end_time)
    return interval_status(start_at, end_at, now)


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
    if has_ended is False:
        return "exclude events known to have ended"
    return "include events in the requested range"


def _ended_filter(
    starts_after: datetime | None,
    starts_before: datetime | None,
    now: datetime,
) -> bool | None:
    if not starts_after and not starts_before:
        return False
    return True if starts_before and starts_before <= now else None


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
    distance = (
        f"; {ev.distance_miles:.2f} miles from the resolved location"
        if ev.distance_miles is not None
        else ""
    )
    if ev.source == "Web discovery":
        source = f"{source}; {_TIER_LABELS.get(ev.tier, ev.tier)}"
        if not ev.structured_source:
            match_warning = "; unconfirmed lead, not a matching option until its constraints are checked"
    return (
        f"- {ev.name}{where}, {when}{timing}{end}{free}{audience}{registration}{status}"
        f"{match_warning}"
        f"{distance}{accessibility} "
        f"({source}) {{cite:{cite}}}{evidence}{details}"
    )


def _explicitly_free(events: list[Event], cost: Literal["free"] | None) -> list[Event]:
    if cost != "free":
        return events
    return [
        event for event in events
        if event.free_evidence or re.search(r"\bfree\b", event.name, re.IGNORECASE)
    ]


async def _handler(args: dict, ctx: ToolContext) -> str:
    query = EventQuery.model_validate(args)
    keyword = (query.topic or "").strip() or None
    borough = (query.borough or "").strip().lower()
    audience = (query.audience or "").strip().lower()
    setting = (query.setting or "").strip().lower()
    resident_requested_count = query.max_results is not None
    result_budget = query.max_results
    near = (query.near or "").strip()
    if near.casefold() in {"nyc", "new york", "new york city"}:
        near = ""
    current_time = datetime.now(NYC_TZ)
    now = query.active_at.astimezone(NYC_TZ) if query.active_at else current_time
    origin = None
    if near:
        stored_origin = current_resolved_location(near, ctx)
        resident_near = resident_supplied_location(
            near, ctx.query, ctx.user_turns, allow_prior=True,
        ) if ctx.query else near
        near = resident_near or (stored_origin.resident_query if stored_origin else "")
        if not near:
            return "A resident-provided NYC location is required before ranking nearby events."
        origin = await resolve_location(near, ctx)
        if origin is None:
            return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
        if origin.low_confidence:
            return f"'{near}' may match several places. Ask for a specific NYC address or landmark."
        if origin.resident_query:
            ctx.current_location = origin

    today = now.strftime("%Y-%m-%d")

    if query.active_at:
        window_start = window_end = now.date().isoformat()
        start_time_bound = end_time_bound = None
        has_started_constraint, has_ended = True, False
    else:
        starts_after = query.starts_after.astimezone(NYC_TZ) if query.starts_after else None
        starts_before = (
            query.starts_before.astimezone(NYC_TZ) if query.starts_before else None
        )
        inclusive_end = starts_before - timedelta(microseconds=1) if starts_before else None
        window_start = (starts_after.date() if starts_after else current_time.date()).isoformat()
        window_end = (inclusive_end.date() if inclusive_end else date.fromisoformat(window_start)).isoformat()
        start_time_bound = (
            starts_after.time().replace(tzinfo=None)
            if starts_after and starts_after.time().replace(tzinfo=None) != time.min
            else None
        )
        end_time_bound = (
            inclusive_end.time().replace(tzinfo=None)
            if inclusive_end and starts_before.time().replace(tzinfo=None) != time.min
            else None
        )
        has_started_constraint = None
        has_ended = _ended_filter(starts_after, starts_before, current_time)
    temporal_instruction = _temporal_instruction(has_started_constraint, has_ended)

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
        if window_start != today or has_ended or has_started_constraint is True
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
    async def ticketmaster_source():
        return await ticketmaster_events(
            keyword=ticketmaster_keyword,
            classification=None,
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
    source_tasks = {
        name: asyncio.create_task(call) for name, call in sources
    }
    if ctx.event_retrieval_policy == "deep":
        done, pending = await asyncio.wait(source_tasks.values(), timeout=60)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results = {
            name: (
                task.result()
                if task in done and not task.cancelled() and task.exception() is None
                else TimeoutError("deep retrieval deadline")
            )
            for name, task in source_tasks.items()
        }
    else:
        gathered = await asyncio.gather(*source_tasks.values(), return_exceptions=True)
        results = dict(zip(source_tasks, gathered))
    unavailable_catalog = [
        name for name in catalog_sources if isinstance(results[name], BaseException)
    ]
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
    def _window_filter(rows: list[Event]) -> list[Event]:
        incomplete = [event for event in rows if not event.start_date]
        kept = _future_only([event for event in rows if event.start_date], window_start)
        if window_end:
            kept = [e for e in kept if e.start_date <= window_end]
        if ctx.event_turn != "preparation" or query.active_at is not None:
            kept = _temporal_filter(
                kept, has_started=has_started_constraint, has_ended=has_ended, now=now,
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
        return kept

    events = _window_filter(events)
    event_result_limit = result_budget
    location_note = ""
    if origin is not None:
        located = [
            event for event in events
            if event.lat is not None and event.lon is not None
        ]
        unlocated = [
            event for event in events
            if event.lat is None or event.lon is None
        ]
        ranked = rank_nearby(
            origin,
            located,
            key=lambda event: event.url or (
                event.name.casefold(), event.start_date, event.start_time, event.venue.casefold(),
            ),
        )
        for event, distance_m in ranked:
            event.distance_miles = miles(distance_m)
        events = [event for event, _distance_m in ranked]
        events += _dedupe_order(unlocated, None)
        location_note = f"Distance from {origin.label}: located listings are ranked nearest first.\n"
        location_note += (
            "Preserve this returned order in the answer; do not reorder by source type.\n"
        )
        source_counts = {
            source: sum(event.source == source for event in located)
            for source in sorted({event.source for event in located})
        }
        location_note += f"Matching records with source coordinates: {source_counts or 'none'}.\n"
        if unlocated:
            location_note += (
                "Only listings whose source provided coordinates were distance-ranked; remaining "
                "leads are not distance-ranked.\n"
            )
    else:
        events = _dedupe_order(events, None)
    complete_events = [
        event for event in events
        if event.start_date and event.name and event.url
        and (event.source != "Web discovery" or event.structured_source)
    ]
    incomplete_events = [event for event in events if event not in complete_events]
    selected_events = (
        complete_events[:event_result_limit]
        if origin is not None and event_result_limit is not None
        else complete_events
        if origin is not None
        else _dedupe_order(complete_events, event_result_limit)
    )
    events = selected_events
    ctx.tool_runs.append({
        "operation": "event_candidate_selection",
        "policy": ctx.event_retrieval_policy,
        "catalog_candidates": len(complete_events),
        "unresolved_leads": len(incomplete_events),
        "eligible_candidates": len(complete_events),
        "selected_candidates": len(selected_events),
        "selected_source_mix": {
            source: sum(event.source == source for event in selected_events)
            for source in sorted({event.source for event in selected_events})
        },
        "stopping_reason": (
            "resident_count" if event_result_limit is not None else "all_eligible_records"
        ),
    })
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
    count_instruction = (
        f"Return no more than {query.max_results} event choices. "
        if resident_requested_count
        else (
            "Choose only the strongest distinct matches for a concise conversational shortlist, "
            "and make clear that more can be listed. "
        )
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
        return setting_note + no_results
    blocks: list[tuple[str, str]] = []
    for ev in events:
        if ev.source == "Web discovery" and not ev.structured_source:
            source_record = ev.provider_record if isinstance(ev.provider_record, dict) else {}
            cite = ctx.citations.register(
                ev.url,
                snippet=ev.evidence_excerpt,
                title=ev.name,
                kind="WEB",
                provenance=source_record.get("provenance") or {
                    "evidence_grade": "discovery",
                    "source_tier": ev.tier,
                },
            )
            blocks.append((cite, _event_block(ev, cite, now)))
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
        if origin is not None and ev.distance_miles is not None:
            derivation["location"] = {
                "origin": [origin.lat, origin.lon],
                "origin_query": near,
                "origin_label": origin.label,
                "point": [ev.lat, ev.lon],
                "distance_mi": ev.distance_miles,
            }
        if ev.structured_source and ev.citation_id:
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
        ev.citation_id = cite
        blocks.append((cite, _event_block(ev, cite, now)))

    available = ctx.evidence_token_budget
    packed_blocks = []
    for citation_id, block in blocks:
        needed = _text_tokens(block, ctx.evidence_model)
        if available is not None and needed > available:
            continue
        packed_blocks.append(block)
        if available is not None:
            available -= needed
    if ctx.evidence_token_budget is not None:
        ctx.evidence_tokens_used += ctx.evidence_token_budget - available
        ctx.evidence_token_budget = available
    packing_note = (
        f"{len(blocks) - len(packed_blocks)} additional ranked event records did not fit in the "
        "current model context; offer a narrower follow-up to inspect them.\n"
        if len(packed_blocks) < len(blocks)
        else ""
    )

    window = f" for {window_start} through {window_end}" if window_end else ""
    free_scope = " whose official source evidence says free" if query.cost == "free" else ""
    header = (
        f"{setting_note}{location_note}{coverage_note}{packing_note}"
        f"NYC event candidates and grounding evidence{window}{free_scope} from live structured "
        "catalogs:\n"
    )
    catalog = header + "\n".join(packed_blocks) if packed_blocks else no_results
    followup = (
        "\nThis is a shortlist, not every matching event. Offer to narrow or list more."
        if not resident_requested_count
        else ""
    )
    diversity_instruction = (
        "Source diversity has already been applied. Preserve the returned order exactly and do "
        "not move a source type ahead of a closer result."
        if origin is not None
        else (
            "For general discovery, include a matching non-marketplace candidate when available "
            "instead of returning only marketplace candidates."
        )
    )
    return (
        f"These candidates form one shared ranked candidate set. Use it in order and "
        f"{temporal_instruction}. Choose by exact date and topic match first. "
        f"{count_instruction}"
        "An exact-date editorial listing excerpt supports only the fields it states. "
        "Normalized candidates already include server-computed temporal status. For an "
        "unstructured web lead, state only the source-backed date and time rather than inferring "
        "a current status. "
        "Keep an incomplete lead only with its specific limitation. "
        f"{diversity_instruction}:\n"
        + catalog
        + followup
    )


def _event_start_at(event: Event) -> datetime | None:
    start_date = _iso_date(event.start_date)
    if not start_date:
        return None
    start_time = _parse_start_time(event.start_time)
    return nyc_datetime(date.fromisoformat(start_date), start_time or time.min)


def _event_fields_supported(event: Event, citation: dict) -> bool:
    evidence = citation_evidence(citation) or ""
    excerpt = " ".join(event.evidence_excerpt.split())
    normalized = " ".join(evidence.casefold().split())
    if not excerpt or excerpt.casefold() not in normalized:
        return False
    normalized = excerpt.casefold()
    for value in (
        event.name,
        event.venue,
        event.borough,
        event.free_evidence,
        event.audience,
        event.accessibility_info,
        event.registration_info,
    ):
        if value and " ".join(value.casefold().split()) not in normalized:
            return False
    claim = "; ".join(filter(None, (
        event.name,
        event.start_date,
        event.start_time,
        event.end_date,
        event.end_time,
        event.venue,
        event.free_evidence,
    ))) + f" {{cite:{event.citation_id}}}"
    scoped_citation = {
        **citation,
        "snippet": excerpt,
        "title": "",
        "provenance": {},
    }
    result = check_grounding(claim, {event.citation_id: scoped_citation})
    return result is None or result.passed


async def _extract_events_handler(args: ExtractEventsInput, ctx: ToolContext) -> str:
    request = ExtractEventsInput.model_validate(args)
    citations = ctx.citations.mapping()
    events: list[Event] = []
    limitations: list[str] = []

    for citation_id in ctx.citations.touched_ids():
        citation = citations.get(citation_id) or {}
        for value in (citation.get("provenance") or {}).get("structured_data") or ():
            events.extend(
                event
                for node in _schema_nodes(value)
                if (event := _schema_event(node, citation_id, citation)) is not None
            )

    for event in request.events:
        citation = citations.get(event.citation_id)
        if citation is None:
            limitations.append(f"{event.name}: source citation was not available")
            continue
        source_url = str(citation.get("url") or "")
        if not _event_fields_supported(event, citation):
            limitations.append(
                f"{event.name}: extracted fields could not be supported by the cited page, "
                f"check {source_url}"
            )
            continue
        ctx.citations.touch(event.citation_id)
        event.url = source_url
        event.source = "Web discovery"
        event.tier = str((citation.get("provenance") or {}).get("source_tier") or "unverified")
        event.provider_id = event.url
        event.provider_record = citation
        event.structured_source = False
        events.append(event)

    after = request.starts_after.astimezone(NYC_TZ) if request.starts_after else None
    before = request.starts_before.astimezone(NYC_TZ) if request.starts_before else None
    selected: list[Event] = []
    for event in _dedupe_order(events, None):
        if _nyc_locality_borough(event.borough) is None:
            limitations.append(
                f"{event.name}: source did not establish an NYC location, check {event.url}"
            )
            continue
        start = _event_start_at(event)
        if start is None:
            limitations.append(f"{event.name}: source did not establish a valid start date")
            continue
        if not event.start_time and any(
            bound is not None and bound.timetz().replace(tzinfo=None) != time.min
            for bound in (after, before)
        ):
            limitations.append(f"{event.name}: source did not establish a start time")
            continue
        if after is not None and start < after:
            continue
        if before is not None and start >= before:
            continue
        if request.active_at is not None and _event_temporal_status(
            event, request.active_at.astimezone(NYC_TZ),
        ) != "in_progress":
            continue
        if request.audience == "kids" and "kids" not in event.audience.casefold():
            continue
        if request.cost == "free" and not event.free_evidence:
            continue
        selected.append(event)

    selected = _dedupe_order(selected, request.max_results)
    now = datetime.now(NYC_TZ)
    confirmed_blocks = [
        _event_block(event, event.citation_id, now)
        for event in selected
        if event.citation_id and event.structured_source
    ]
    lead_blocks = [
        _event_block(event, event.citation_id, now)
        for event in selected
        if event.citation_id and not event.structured_source
    ]
    if not confirmed_blocks:
        result = "No source-backed events matched the requested hard constraints."
    else:
        result = "Source-backed events after deterministic date and time checks:\n" + "\n".join(
            confirmed_blocks
        )
    if lead_blocks:
        result += (
            "\nSource-reported leads whose extracted fields still need confirmation:\n"
            + "\n".join(lead_blocks)
        )
    if limitations:
        result += "\nIncomplete leads:\n- " + "\n- ".join(limitations)
    return result


def extract_events_tool() -> Tool:
    return Tool(
        name="extract_events",
        description="Validate and time-filter events extracted from fetched sources",
        input_type=ExtractEventsInput,
        handler=_extract_events_handler,
        title="Extract events",
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_nyc_events",
            description=(
                "Structured NYC event catalogs. Pass absolute New York times and stated topic, "
                "place, audience, setting, cost, or explicit count. Use `active_at` only when the "
                "event must be underway then."
            ),
            input_type=EventQuery,
            handler=_handler,
            open_world=True,  # hits live Ticketmaster + Socrata
            title="Find NYC events",
        ),
        extract_events_tool(),
    ]
