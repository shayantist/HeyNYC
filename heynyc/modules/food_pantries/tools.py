"""food_pantries module tool: `nearest_food_pantry`, grounded in the city's FoodHelp backend.

Data source: the public, tokenless ArcGIS Feature Service that powers finder.nyc.gov/foodhelp
(Food_Help_Programs_PROD_view, ~522 open sites). We fetch the whole layer once (generic ArcGIS
adapter), rank sites by Haversine distance from the user's geocoded location (reused geo machinery),
and return the closest few with: name, full address, open-now status (computed from the structured
fp_<day>_open*/close* hours), phone, dietary/access flags (Halal/Kosher/HIV/Mobile), and a Google
Maps directions link. Every site is a row-addressed DATA citation grounded in the ArcGIS source URL.

Honest limitations (enforced in the manifest prompt too): the source has NO languages field, and
`fp_notes` (eligibility) is frequently blank, we never invent hours, requirements, or languages.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    _clarify_message,
    _resolution_note,
    format_distance,
    geocode,
    haversine_m,
    miles,
    resident_supplied_location,
)

# The live backend of finder.nyc.gov/foodhelp, verified public + tokenless.
FOODHELP_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Food_Help_Programs_PROD_view/FeatureServer/0"
)
WHERE_OPEN = "status='Open'"
FOODHELP_QUERY_URL = (
    f"{FOODHELP_URL}/query?"
    + urlencode(
        {
            "where": WHERE_OPEN,
            "outFields": "*",
            "f": "geojson",
            "resultRecordCount": 2000,
        }
    )
)
OFFICIAL = "finder.nyc.gov/foodhelp or call 311"
NO_LOCATION = (
    "The proposed search origin was not supplied by the user, so do not use it. "
    "For immediate food help, tell the user to call 311 or use "
    "https://finder.nyc.gov/foodhelp. Also ask where they are, using an NYC address or "
    "neighborhood, so the next search can return nearby options. Never guess their location."
)
SOURCE_LOCATION_NEEDS_FETCH = (
    "The cited search result is not enough to establish that origin. First call official_sources on "
    "the cited page for a source span containing both the resident-named place and exact address, "
    "then retry with its new citation id. If the fetched page does not contain both together, ask "
    "the resident for their NYC address or neighborhood."
)
SOURCE_LOCATION_NEEDS_CONFIRMATION = (
    "The authoritative source did not establish that the proposed address belongs to the "
    "resident-named place. Stop calling tools for this location and ask the resident for their "
    "exact NYC address or intersection. Do not repeat the unverified proposed address or mention "
    "tool, source-validation, or internal safety mechanics in the resident answer."
)
_NYC_TZ = ZoneInfo("America/New_York")

# datetime.weekday(): Monday=0 … Sunday=6 → the source's fp_<day>_* prefixes.
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
# Dietary / access descriptors decoded from the `type_fp` / `type_sk` coded-value domains
# (verified against the live FeatureServer layer definition). FP / SK are the plain base
# types → no descriptor.
_TYPE_DESCRIPTORS = {
    "FPH": "Halal", "FPHA": "HIV Customers", "FPK": "Kosher", "FPM": "Mobile",
    "SKK": "Kosher", "SKM": "Mobile",
}
def _clean(value) -> str:
    """None / literal 'NULL' / blanks → ''. ArcGIS returns JSON null for empty fields."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() == "NULL" else text


def _resident_supplied_origin(near: str, query: str, user_turns: tuple[str, ...]) -> str:
    # F159: took `user_turns`, passed `()`, so only the current message was searchable
    # A confirmation carries no address, so an earlier-turn location was unreachable
    # Widens WHERE we look, not WHAT counts: still resident-authored only
    current_turn = query or (user_turns[-1] if user_turns else "")
    if not current_turn:
        return near
    return resident_supplied_location(
        near, current_turn, tuple(user_turns), allow_prior=True
    )


def _source_span_supports_origin(sentence: str, place: str, address: str) -> bool:
    def normalized(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(
            "".join(
                " " if unicodedata.category(character)[0] in {"P", "Z"} else character
                for character in value
            ).split()
        )

    sentence = normalized(sentence)
    place = normalized(place)
    address = normalized(address)
    place_start = sentence.find(place)
    address_start = sentence.find(address)
    if place_start < 0 or address_start < 0:
        return False
    if place_start < address_start:
        between = sentence[place_start + len(place):address_start]
    else:
        between = sentence[address_start + len(address):place_start]
    return not any(character.isdigit() for character in between)


def _source_origin(
    near: str,
    source_place: str,
    source_id: str,
    ctx: ToolContext,
) -> tuple[str, str]:
    resident_place = resident_supplied_location(source_place, ctx.query, ())
    if not resident_place:
        return "", source_id
    citations = ctx.citations.mapping()
    candidates = (
        [(source_id, citations.get(source_id, {}))]
        if source_id
        else list(citations.items())
    )
    matches = [
        (candidate_id, citation)
        for candidate_id, citation in candidates
        if any(
            _source_span_supports_origin(sentence, resident_place, near)
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n+",
                citation.get("snippet", ""),
            )
        )
    ]
    authoritative = [
        (candidate_id, citation)
        for candidate_id, citation in matches
        if (citation.get("provenance") or {}).get("evidence_grade") == "authoritative"
    ]
    if len(authoritative) == 1:
        return near, authoritative[0][0]
    if matches:
        return "", matches[0][0]
    return "", source_id


# --- hours / open-now ------------------------------------------------------

_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$")
_MONTHLY_OCCURRENCE_RE = re.compile(r"\b([1-5])(?:st|nd|rd|th)?\b", re.IGNORECASE)


def _parse_time(value) -> int | None:
    """A source time string → minutes since midnight, or None if blank/unparseable.

    Defensive across the formats the finder might emit: '9:00 AM', '5:30 PM', 24h '17:30',
    and military '0900'. NOTE: the exact live format is not verified offline, if the pull shows
    a different encoding, extend this parser (the open-now display fails safe to 'hours not listed').
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return None
    if text.isdigit() and len(text) in (3, 4):          # military 'HMM' / 'HHMM'
        hour, minute = divmod(int(text), 100)
        return hour * 60 + minute if 0 <= hour <= 23 and 0 <= minute <= 59 else None
    match = _TIME_RE.match(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if meridiem:
        hour = 0 if hour == 12 else hour
        if meridiem == "pm":
            hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _prefix(record: dict) -> str:
    """Soup kitchens carry sk_<day>_* hours; food pantries carry fp_<day>_*."""
    return {"FP": "fp", "SK": "sk"}.get(
        _clean(record.get("program_type")).upper(),
        "",
    )


def _day_slots(record: dict, prefix: str, day: str) -> list[tuple[int, int]]:
    """The (open, close) minute-ranges for one weekday (up to three windows)."""
    if not prefix:
        return []
    slots: list[tuple[int, int]] = []
    for n in ("1", "2", "3"):
        opened = _parse_time(record.get(f"{prefix}_{day}_open{n}"))
        closed = _parse_time(record.get(f"{prefix}_{day}_close{n}"))
        if opened is not None and closed is not None and closed != opened:
            slots.append((opened, closed))
    return slots


def _open_now(record: dict, now: datetime) -> bool | None:
    """True/False if today's structured hours say so; None when no hours are listed at all.

    Never guesses: a record with hours on other days but none today reads as closed today (False);
    a record with no parseable hours anywhere returns None so the agent says it doesn't have them.
    """
    if _schedule_conflict(record, now.date()):
        return None
    if _monthly_occurrence_allows(record, now.date()) is False:
        return False
    prefix = _prefix(record)
    today_slots = _day_slots(record, prefix, _DAYS[now.weekday()])
    previous_slots = _day_slots(record, prefix, _DAYS[(now.weekday() - 1) % 7])
    minutes = now.hour * 60 + now.minute
    if today_slots:
        if any(
            (open_m < close_m and open_m <= minutes < close_m)
            or (open_m > close_m and minutes >= open_m)
            for open_m, close_m in today_slots
        ):
            return True
    if any(open_m > close_m and minutes < close_m for open_m, close_m in previous_slots):
        return True
    if any(_day_slots(record, prefix, day) for day in _DAYS):
        return False
    return None


def _scheduled_on(record: dict, requested: date) -> bool | None:
    if _schedule_conflict(record, requested):
        return None
    if _monthly_occurrence_allows(record, requested) is False:
        return False
    prefix = _prefix(record)
    if _day_slots(record, prefix, _DAYS[requested.weekday()]):
        return True
    if any(_day_slots(record, prefix, day) for day in _DAYS):
        return False
    return None


def _scheduled_during(
    record: dict,
    requested: date,
    window_start: int,
    window_end: int,
) -> bool | None:
    if _schedule_conflict(record, requested):
        return None
    if _monthly_occurrence_allows(record, requested) is False:
        return False
    prefix = _prefix(record)
    slots = _day_slots(record, prefix, _DAYS[requested.weekday()])
    if slots:
        return any(
            opened < window_end and (closed if closed > opened else closed + 1440) > window_start
            for opened, closed in slots
        )
    if any(_day_slots(record, prefix, day) for day in _DAYS):
        return False
    return None


def _schedule_conflict(record: dict, requested: date) -> bool:
    prefix = _prefix(record)
    source_days = _clean(record.get(f"{prefix}_days_orig"))
    notes = _clean(record.get(f"{prefix}_notes"))
    if (
        _DAYS[requested.weekday()].upper() not in source_days.upper()
        or requested.strftime("%A").lower() not in notes.lower()
    ):
        return False
    source_occurrences = set(_MONTHLY_OCCURRENCE_RE.findall(source_days))
    note_occurrences = set(_MONTHLY_OCCURRENCE_RE.findall(notes))
    return bool(source_occurrences and note_occurrences and source_occurrences != note_occurrences)


def _monthly_occurrence_allows(record: dict, requested: date) -> bool | None:
    prefix = _prefix(record)
    day = _DAYS[requested.weekday()]
    weekday = requested.strftime("%A").lower()
    occurrence_sets = [
        set(_MONTHLY_OCCURRENCE_RE.findall(text))
        for text in (
            _clean(record.get(f"{prefix}_days_orig")),
            _clean(record.get(f"{prefix}_notes")),
        )
        if day in text.lower() or weekday in text.lower()
    ]
    occurrence_sets = [values for values in occurrence_sets if values]
    if not occurrence_sets or any(values != occurrence_sets[0] for values in occurrence_sets[1:]):
        return None
    occurrence = str((requested.day - 1) // 7 + 1)
    return occurrence in occurrence_sets[0]


def _status_label(open_now: bool | None) -> str:
    if open_now is True:
        return "scheduled open now"
    if open_now is False:
        return "scheduled closed now"
    return "hours not listed, call ahead"


def _listed_hours(record: dict, weekday: int) -> str:
    prefix = _prefix(record)
    day = _DAYS[weekday]
    slots = []
    for n in ("1", "2", "3"):
        opened = _clean(record.get(f"{prefix}_{day}_open{n}"))
        closed = _clean(record.get(f"{prefix}_{day}_close{n}"))
        if opened and closed:
            slots.append(f"{opened}-{closed}")
    return ", ".join(slots)


# --- record → pantry -------------------------------------------------------

@dataclass
class FoodPantry:
    name: str
    lat: float
    lon: float
    address: str
    phone: str
    type_fp: str
    type_sk: str
    notes: str
    global_id: str
    valid_as_of: str
    raw: dict = field(default_factory=dict)


def _address(record: dict) -> str:
    """Assemble the street address from the source's parts (any of which may be blank)."""
    line1 = ", ".join(p for p in (_clean(record.get("distadd")),
                                  _clean(record.get("dist_location_info"))) if p)
    tail = " ".join(p for p in (_clean(record.get("distboro")),
                                _clean(record.get("distzip"))) if p)
    return ", ".join(p for p in (line1, tail) if p)


def _epoch_ms_to_date(ms: float) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return ""


def _valid_as_of(record: dict) -> str:
    """The record's EditDate (ArcGIS epoch-ms or ISO), or blank when the source omits it."""
    value = record.get("EditDate")
    if isinstance(value, (int, float)):
        parsed = _epoch_ms_to_date(value)
    else:
        text = _clean(value)
        if text.isdigit():
            parsed = _epoch_ms_to_date(int(text))
        else:
            try:
                parsed = datetime.fromisoformat(text[:10]).date().isoformat()
            except ValueError:
                parsed = ""
    return parsed


def _to_pantry(record: dict) -> FoodPantry | None:
    """Map a raw feature record to a FoodPantry; drop records without usable coordinates."""
    try:
        lat = float(record["lat"])
        lon = float(record["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return FoodPantry(
        name=_clean(record.get("program")) or "Food pantry",
        lat=lat,
        lon=lon,
        address=_address(record),
        phone=_clean(record.get("org_phone")),
        type_fp=_clean(record.get("type_fp")),
        type_sk=_clean(record.get("type_sk")),
        notes=_clean(record.get("fp_notes")),
        global_id=_clean(record.get("GlobalID")),
        valid_as_of=_valid_as_of(record),
        raw=record,
    )


def _flags(pantry: FoodPantry) -> list[str]:
    """Dietary/access descriptors decoded from the `type_fp` / `type_sk` coded-value domains
    (Halal/Kosher/HIV Customers/Mobile); [] for a plain FP/SK base type."""
    flags: list[str] = []
    for code in (pantry.type_fp, pantry.type_sk):
        descriptor = _TYPE_DESCRIPTORS.get(code.strip().upper())
        if descriptor and descriptor not in flags:
            flags.append(descriptor)
    return flags


def directions_link(lat: float, lon: float) -> str:
    """A Google Maps directions deep link to a grounded coordinate (navigation handoff, no citation
    needed, it's a deterministic transform of an already-grounded point)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.5f},{lon:.5f}"


# --- the tool --------------------------------------------------------------

def _pantry_citation(ctx: ToolContext, pantry: FoodPantry, *,
                     origin_lat: float, origin_lon: float, dist_mi: float) -> str:
    """Register a row-addressed DATA citation: the single-feature ArcGIS permalink, the row
    snapshot + content hash, and the distance derivation (so the eval floor can recompute it)."""
    url = (feature_query_url(FOODHELP_URL, pantry.global_id, id_field="GlobalID")
           if pantry.global_id else FOODHELP_URL)
    provenance = data_provenance(
        pantry.raw,
        record_id=pantry.global_id,
        field_pointer="/",
        derivation={"origin": [origin_lat, origin_lon], "point": [pantry.lat, pantry.lon],
                    "distance_mi": dist_mi, "temporal_basis": "weekly_schedule"},
    )
    return ctx.citations.register(
        url,
        snippet=f"{pantry.name}, {pantry.address}",
        title="NYC FoodHelp (Food Help Programs)",
        kind="DATA",
        valid_as_of=pantry.valid_as_of,
        provenance=provenance,
    )


def _availability_citation(
    ctx: ToolContext,
    *,
    pantries: list[FoodPantry],
    nearby: list[FoodPantry],
    now: datetime,
    requested: date | None = None,
    service_window: tuple[int, int] | None = None,
) -> str:
    if service_window:
        requested = requested or now.date()
        start, end = service_window
        scheduled_open_nearby = sum(
            _scheduled_during(pantry.raw, requested, start, end) is True
            for pantry in nearby
        )
        scheduled_open_citywide = sum(
            _scheduled_during(pantry.raw, requested, start, end) is True
            for pantry in pantries
        )
        basis = (
            f"weekly hours overlapping "
            f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"
        )
    else:
        scheduled_open_nearby = sum(_open_now(pantry.raw, now) is True for pantry in nearby)
        scheduled_open_citywide = sum(_open_now(pantry.raw, now) is True for pantry in pantries)
        basis = "weekly hours indicating open"
    snapshot = {
        "lookup_at": now.isoformat(),
        "status_filter": WHERE_OPEN,
        "nearby_records_checked": len(nearby),
        "citywide_records_checked": len(pantries),
        "scheduled_open_nearby": scheduled_open_nearby,
        "scheduled_open_citywide": scheduled_open_citywide,
    }
    return ctx.citations.register(
        FOODHELP_QUERY_URL,
        snippet=(
            f"At lookup, {scheduled_open_nearby} of {len(nearby)} nearby and "
            f"{scheduled_open_citywide} of {len(pantries)} City-listed sites had {basis}"
        ),
        title="NYC FoodHelp availability lookup",
        kind="DATA",
        valid_as_of="",
        provenance=data_provenance(
            snapshot,
            record_id="availability-summary",
            field_pointer="/",
            derivation={
                "temporal_basis": "weekly_schedule",
                "response_priority_anchors": ["311", OFFICIAL],
            },
        ),
    )


def _pantry_block(
    pantry: FoodPantry,
    cite: str,
    distance: str,
    now: datetime,
    requested: date | None = None,
    service_window: tuple[int, int] | None = None,
) -> str:
    flags = _flags(pantry)
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    show_hours = _monthly_occurrence_allows(
        pantry.raw, requested or now.date()
    ) is not False
    if service_window:
        requested = requested or now.date()
        start, end = service_window
        scheduled = _scheduled_during(pantry.raw, requested, start, end)
        status = (
            f"weekly schedule overlaps the requested "
            f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d} service window"
            if scheduled is True
            else "weekly schedule does not overlap the requested service window"
            if scheduled is False
            else "requested-window hours are not established, call ahead"
        )
        weekday = requested.weekday()
    elif requested and requested != now.date():
        day = requested.strftime("%A, %Y-%m-%d")
        if _schedule_conflict(pantry.raw, requested):
            status = (
                f"source fields conflict about the {requested.strftime('%A')} schedule "
                f"for {day}, call to verify"
            )
        elif _monthly_occurrence_allows(pantry.raw, requested) is False:
            status = f"monthly schedule excludes {day}, call to verify changes"
        else:
            scheduled = _scheduled_on(pantry.raw, requested)
            status = (
                f"weekly schedule lists hours on {day}"
                if scheduled is True
                else f"weekly schedule lists no hours on {day}"
                if scheduled is False
                else f"hours not listed for {day}, call ahead"
            )
        weekday = requested.weekday()
    else:
        status = _status_label(_open_now(pantry.raw, now))
        weekday = now.weekday()
    parts = [f"- {pantry.name}{flag_str} ({pantry.address or 'NYC'}), "
             f"{distance}, {status} {{cite:{cite}}}"]
    hours = _listed_hours(pantry.raw, weekday)
    if hours and show_hours:
        label = (
            f"{requested.strftime('%A')}'s"
            if requested and requested != now.date()
            else "Today's"
        )
        parts.append(f"  {label} listed weekly hours: {hours}")
    if pantry.phone:
        parts.append(f"  Phone: {pantry.phone}")
    if pantry.notes:
        parts.append(f"  Eligibility/notes: {pantry.notes}")
    parts.append(f"  Directions: {directions_link(pantry.lat, pantry.lon)}")
    parts.append(f"  As of: {pantry.valid_as_of or 'Source date unavailable'}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    if not near:
        return NO_LOCATION
    service_type = str(args.get("service_type") or "any")
    if service_type not in {"pantry", "soup_kitchen", "any"}:
        return "The requested service type is invalid. Use pantry, soup_kitchen, or any."
    service_name = {
        "pantry": "food pantry",
        "soup_kitchen": "soup kitchen",
        "any": "food-help site",
    }[service_type]
    now = datetime.now(_NYC_TZ)
    on = str(args.get("on") or "").strip()
    try:
        requested = date.fromisoformat(on) if on else None
    except ValueError:
        return "The requested date is invalid. Ask for a date in YYYY-MM-DD format."
    if requested and requested < now.date():
        return (
            "NYC FoodHelp provides current weekly schedules, so I cannot verify a past "
            "service date. Ask for today or a future date."
        )
    window_start = str(args.get("service_window_start") or "").strip()
    window_end = str(args.get("service_window_end") or "").strip()
    if bool(window_start) != bool(window_end):
        return "Pass both service_window_start and service_window_end, or omit both."
    service_window = None
    if window_start:
        start = _parse_time(window_start)
        end = _parse_time(window_end)
        if start is None or end is None or end <= start:
            return (
                "The requested service window is invalid. Use same-day 24-hour HH:MM values "
                "with the end after the start."
            )
        service_window = (start, end)
    future_schedule = requested is not None and requested > now.date()
    requested_day = requested or now.date()
    urgent = args.get("urgent") is True and not future_schedule
    if args.get("urgent") is True and not future_schedule and service_window is None:
        return (
            "urgent=true requires service_window_start and service_window_end. Preserve the "
            "resident's requested same-day timeframe and pass it as NYC-local 24-hour HH:MM."
        )
    source_id = str(args.get("near_source_citation") or "")
    source_place = str(args.get("near_source_place") or "")
    resident_origin = _resident_supplied_origin(near, ctx.query, ctx.user_turns)
    source_expanded_origin = bool(
        (source_id or source_place)
        and resident_origin
        and unicodedata.normalize("NFKC", resident_origin).casefold()
        != unicodedata.normalize("NFKC", near).casefold()
    )
    source_origin, source_id = (
        _source_origin(near, source_place, source_id, ctx)
        if not resident_origin or source_expanded_origin
        else ("", source_id)
    )
    if source_expanded_origin:
        resident_origin = ""
    if source_id and not resident_origin and not source_origin:
        citation = ctx.citations.mapping().get(source_id, {})
        if (citation.get("provenance") or {}).get("evidence_grade") == "authoritative":
            return f"{SOURCE_LOCATION_NEEDS_CONFIRMATION} Source: {{cite:{source_id}}}."
        return f"{SOURCE_LOCATION_NEEDS_FETCH} Source: {{cite:{source_id}}}."
    resident_origin = resident_origin or source_origin
    if not resident_origin:
        return NO_LOCATION
    near = resident_origin

    origin = await geocode(near, client=ctx.http)
    if origin is None:
        source_marker = f" The proposed address came from {{cite:{source_id}}}." if source_origin else ""
        return (
            f"I couldn't locate '{near}' in NYC, so I can't find a nearby {service_name}. Ask the "
            f"user for a specific NYC address or neighborhood, don't guess a {service_name}."
            f"{source_marker}"
        )
    if origin.low_confidence:
        return _clarify_message(near)

    try:
        records = await query_feature_service(FOODHELP_URL, where=WHERE_OPEN, client=ctx.http)
    except httpx.HTTPError:
        return (
            f"I couldn't reach the city's FoodHelp data right now, don't guess a {service_name}. "
            f"Point the user to {OFFICIAL}."
        )

    pantries = [p for p in (_to_pantry(r) for r in records) if p is not None]
    if service_type != "any":
        expected_prefix = "fp" if service_type == "pantry" else "sk"
        pantries = [pantry for pantry in pantries if _prefix(pantry.raw) == expected_prefix]
    if not pantries:
        result_label = {
            "pantry": "food pantries",
            "soup_kitchen": "soup kitchens",
            "any": "food-help sites",
        }[service_type]
        cite = _availability_citation(
            ctx,
            pantries=[],
            nearby=[],
            now=now,
        )
        if urgent:
            ctx.response_priority_citation_ids.add(cite)
        return (
            f"No open {result_label} came back from the city's FoodHelp data. Don't invent one, "
            f"point the user to {OFFICIAL}. {{cite:{cite}}}"
        )
    k = int(args.get("k") or 5)
    ordered = sorted(pantries, key=lambda p: haversine_m(origin.lat, origin.lon, p.lat, p.lon))
    # Collapse duplicate rows for the same physical site (same name + coordinate).
    unique: list[FoodPantry] = []
    seen: set[tuple] = set()
    for pantry in ordered:
        key = (pantry.name.strip().lower(), round(pantry.lat, 5), round(pantry.lon, 5))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pantry)
    if service_window:
        def requested_window_rank(pantry: FoodPantry) -> tuple[float, float]:
            scheduled = _scheduled_during(
                pantry.raw,
                requested_day,
                service_window[0],
                service_window[1],
            )
            return (
                0 if scheduled is True else 2 if scheduled is False else 1,
                haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon),
            )

        unique.sort(key=requested_window_rank)
    elif future_schedule:
        def requested_day_rank(pantry: FoodPantry) -> tuple[float, float]:
            scheduled = _scheduled_on(pantry.raw, requested)
            return (
                0 if scheduled is True else 2 if scheduled is False else 1,
                haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon),
            )

        unique.sort(key=requested_day_rank)
    ranked = unique[:k]
    if service_window:
        def availability(pantry: FoodPantry) -> bool | None:
            return _scheduled_during(
                pantry.raw,
                requested_day,
                service_window[0],
                service_window[1],
            )
    else:
        def availability(pantry: FoodPantry) -> bool | None:
            return _open_now(pantry.raw, now)

    scheduled_open = [pantry for pantry in ranked if availability(pantry) is True]
    unknown_hours = [pantry for pantry in ranked if availability(pantry) is None]
    citywide_open = [pantry for pantry in unique if availability(pantry) is True]
    scheduled_open_count = len(scheduled_open)
    citywide_scheduled_open = len(citywide_open)
    citywide_unknown_hours = sum(availability(pantry) is None for pantry in unique)
    availability_citation = (
        _availability_citation(
            ctx,
            pantries=unique,
            nearby=ranked,
            now=now,
            requested=requested_day,
            service_window=service_window,
        )
        if urgent
        else ""
    )
    if availability_citation:
        ctx.response_priority_citation_ids.add(availability_citation)
    availability_marker = (
        f" {{cite:{availability_citation}}}" if availability_citation else ""
    )
    displayed = (
        scheduled_open[:1]
        if urgent and service_window
        else scheduled_open or citywide_open[:1] or unknown_hours
        if urgent
        else ranked
    )

    lines = [
        f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})",
        _resolution_note(near, origin),
        (
            f"City-listed {service_name} evidence from NYC FoodHelp "
            "(finder.nyc.gov/foodhelp):"
            if urgent
            else f"Nearest City-listed {service_name} candidates from NYC FoodHelp "
                 "(finder.nyc.gov/foodhelp), report only these, cite each:"
        ),
    ]
    if source_origin:
        lines.insert(2, f"Origin address came from the official source {{cite:{source_id}}}.")
    if future_schedule:
        lines.append(
            f"Results are ranked for the weekly schedule on "
            f"{requested.strftime('%A, %Y-%m-%d')}; this does not confirm service that day."
        )
    if urgent and service_window and scheduled_open:
        lines.append(
            f"Immediate food need in the requested {window_start}-{window_end} service window: "
            "the listed weekly schedule overlaps that window but does not confirm food "
            "availability. Lead with call 311 or https://finder.nyc.gov/foodhelp for immediate "
            "help, then give the single nearest weekly-schedule lead and tell the resident to "
            f"call the listed site before traveling.{availability_marker}"
        )
    elif urgent and service_window:
        lines.append(
            f"Immediate food need in the requested {window_start}-{window_end} service window: "
            "no City-listed site in this feed has weekly hours establishing service in that "
            "window. Lead with call 311 or https://finder.nyc.gov/foodhelp. Do not present "
            f"another time of day as an option for that window.{availability_marker}"
        )
    elif urgent and scheduled_open:
        lines.append(
            "Immediate food need: a weekly schedule does not confirm food availability now or "
            "later today. Lead with asking the resident to call the listed site now. If the site "
            "cannot confirm service, tell them to call 311 or use "
            f"https://finder.nyc.gov/foodhelp, then offer to search farther.{availability_marker}"
        )
    elif urgent and citywide_scheduled_open:
        lines.append(
            "Immediate food need: none of the nearest candidates is scheduled open now, so do not "
            "show their closed-site cards. Lead with call 311 or "
            "https://finder.nyc.gov/foodhelp, then give the single nearest farther site whose "
            "weekly schedule says open. Label it a farther scheduled-open lead and tell the "
            "resident to call before traveling because the feed does not confirm food "
            f"availability.{availability_marker}"
        )
    elif urgent and unknown_hours:
        lines.append(
            "Immediate food need: the nearest candidates' hours are unavailable, so they may "
            "still be open. Lead with call 311 or https://finder.nyc.gov/foodhelp. The listed "
            "candidates are call-only leads: tell the resident to call now and not travel unless "
            f"a site confirms service.{availability_marker}"
        )
    elif urgent:
        lines.append(
            "Immediate food need: no City-listed site in this feed is scheduled open now. Lead "
            "with call 311 or https://finder.nyc.gov/foodhelp. Do not show closed-site cards or "
            f"offer to search farther in the same feed.{availability_marker}"
        )

    if not future_schedule and scheduled_open_count and not service_window:
        verb = "is" if scheduled_open_count == 1 else "are"
        lines.append(f"{scheduled_open_count} of these candidates {verb} scheduled open now.")
    elif not future_schedule and not urgent:
        lines.append(
            "None of these nearest candidates is scheduled open now. Do not present them as food "
            "available now; offer to search farther or call 311 for immediate food help."
        )
        if citywide_scheduled_open == 0 and citywide_unknown_hours:
            lines.append(
                f"Hours are unavailable for {citywide_unknown_hours} City-listed site"
                f"{'s' if citywide_unknown_hours != 1 else ''}, which may still be open. "
                "Do not call those sites closed; tell the user to call the listed site or 311."
            )
        elif citywide_scheduled_open == 0:
            lines.append(
                "No City-listed site in this feed is scheduled open now. Tell the user to call 311 "
                "for immediate food help; do not offer to search farther in the same feed."
            )
        else:
            lines.append(
                "Lead with call 311 or https://finder.nyc.gov/foodhelp for immediate food help, "
                "then offer to search farther if the resident wants. Do not include a farther "
                "site until they ask to widen the search."
            )
    if urgent and not scheduled_open and citywide_scheduled_open and not service_window:
        lines.append("Farther scheduled-open lead, call before traveling:")
    for pantry in displayed:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon))
        cite = _pantry_citation(ctx, pantry, origin_lat=origin.lat, origin_lon=origin.lon,
                                dist_mi=dist_mi)
        lines.append(
            _pantry_block(
                pantry,
                cite,
                format_distance(near, origin, dist_mi),
                now,
                requested,
                service_window,
            )
        )
    lines.append("The listed weekly schedule has no holiday or temporary exception fields, so "
                 "tell the user to call before leaving. Eligibility isn't always listed; if a "
                 "field isn't shown, say you don't have it. The data has no language info.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="nearest_food_pantry",
            description=(
                "Find the nearest City-listed NYC food pantries / soup kitchens to an address, grounded in "
                "the city's official FoodHelp data (finder.nyc.gov/foodhelp). Pass `near` = the "
                "user's NYC address or neighborhood. If they have not supplied one, ask before "
                "calling and never invent a broad origin. Pass `service_type=pantry` for a pantry "
                "request, `soup_kitchen` for a soup-kitchen request, or `any` for general food "
                "help. Pass `on=YYYY-MM-DD` for a future date and label its results as scheduled "
                "weekly hours, not guaranteed availability; optional `k` defaults to 5. Returns "
                "each site's name, full address, requested-date weekly hours or current-day "
                "scheduled-open status, phone, dietary/access type "
                "(Halal/Kosher/HIV/Mobile), and a Google Maps directions link, every site cited. "
                "Set `urgent=true` when the resident needs food now, today, or tonight so the "
                "result leads with the immediate fallback and does not overstate weekly hours. "
                "When they need service during a named same-day time window, also pass "
                "`service_window_start` and `service_window_end` as 24-hour HH:MM values. For "
                "example, tonight is 17:00-23:59 unless the resident gives narrower times. "
                "NEVER guess a pantry: if geocoding fails or none are near, say so and point to 311. "
                "The source has no language info and eligibility notes are often blank, don't invent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string",
                             "description": "The NYC address or neighborhood to search near."},
                    "near_source_citation": {
                        "type": "string",
                        "pattern": r"^S\d+$",
                        "description": (
                            "An authoritative citation ID whose text contains the exact `near` "
                            "address. Use only when an official source resolved a resident-named "
                            "place that could not itself be geocoded."
                        ),
                    },
                    "near_source_place": {
                        "type": "string",
                        "description": (
                            "The resident's exact named place resolved by near_source_citation. "
                            "The place and `near` address must occur together in that source."
                        ),
                    },
                    "k": {"type": "integer",
                          "description": "How many food-help sites to return (default 5).",
                          "default": 5},
                    "urgent": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "True when the resident needs food immediately, today, or tonight"
                        ),
                    },
                    "on": {
                        "type": "string",
                        "format": "date",
                        "description": (
                            "Requested service date in YYYY-MM-DD. Pass when the resident names "
                            "a specific future date or day."
                        ),
                    },
                    "service_window_start": {
                        "type": "string",
                        "pattern": r"^\d{2}:\d{2}$",
                        "description": (
                            "Start of the resident's requested same-day service window in local "
                            "NYC 24-hour HH:MM time. Pass together with service_window_end."
                        ),
                    },
                    "service_window_end": {
                        "type": "string",
                        "pattern": r"^\d{2}:\d{2}$",
                        "description": (
                            "End of the resident's requested same-day service window in local "
                            "NYC 24-hour HH:MM time. Pass together with service_window_start."
                        ),
                    },
                    "service_type": {
                        "type": "string",
                        "enum": ["pantry", "soup_kitchen", "any"],
                        "default": "any",
                        "description": "The source service type requested by the resident.",
                    },
                },
                "required": ["near"],
            },
            handler=_handler,
            open_world=True,  # hits the live ArcGIS FoodHelp service + geocoder
        )
    ]
