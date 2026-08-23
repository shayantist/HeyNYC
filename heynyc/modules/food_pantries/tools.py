"""food_pantries module tool: `find_foodhelp_locations`, grounded in the city's FoodHelp backend.

Data source: the public, tokenless ArcGIS Feature Service that powers finder.nyc.gov/foodhelp
(Food_Help_Programs_PROD_view, ~522 open sites). We fetch the whole layer once (generic ArcGIS
adapter), rank sites by Haversine distance from the user's geocoded location (reused geo machinery),
and return the closest few with: name, full address, open-now status (computed from the structured
fp_<day>_open*/close* hours), phone, dietary/access flags (Halal/Kosher/HIV/Mobile), and a Google
Maps directions link. Every site is a row-addressed DATA citation grounded in the ArcGIS source URL.

Honest limitations (enforced in the manifest prompt too): the source has no languages or
eligibility fields. Its `fp_notes` field describes schedule recurrence, so we never reinterpret it
as eligibility or invent hours, requirements, or languages.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Literal
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.temporal import parse_clock_minutes, weekly_open_status
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service_page
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    current_resolved_location,
    haversine_m,
    miles,
    origin_precision,
    rank_nearby,
    resident_supplied_location,
    resolve_location,
    resolved_location_citation,
)

# The live backend of finder.nyc.gov/foodhelp, verified public + tokenless.
FOODHELP_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Food_Help_Programs_PROD_view/FeatureServer/0"
)
WHERE_OPEN = "status='Open'"
FOODHELP_PAGE_SIZE = 2000
FOODHELP_QUERY_URL = (
    f"{FOODHELP_URL}/query?"
    + urlencode(
        {
            "where": WHERE_OPEN,
            "outFields": "*",
            "f": "geojson",
            "resultRecordCount": FOODHELP_PAGE_SIZE,
        }
    )
)
FOOD_ROUTE_URL = "https://access.nyc.gov/programs/emergency-food-assistance/"
FOOD_ROUTE_FACT = (
    "Use the Food Help NYC map or call 311 and ask for food locations near you"
)
_NYC_TZ = ZoneInfo("America/New_York")


class FoodHelpWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str = Field(pattern=r"^\d{2}:\d{2}$", description="Window start in 24-hour HH:MM time.")
    end: str = Field(pattern=r"^\d{2}:\d{2}$", description="Window end in 24-hour HH:MM time.")


class FoodHelpQuery(LocationRequest):
    near: str = Field(description="NYC address, neighborhood, or landmark to search near.")
    max_results: int | None = Field(
        default=None, ge=1, le=10, description="Maximum food-help sites requested; omit for the default 5."
    )
    visit_date: date | None = Field(default=None, description="Requested New York visit date.")
    visit_time: time | None = Field(default=None, description="Requested New York visit time.")
    near_source_citation: str | None = Field(
        default=None,
        pattern=r"^S\d+$",
        description=(
            "Authoritative citation containing the exact near address when an official source "
            "resolved a resident-named place that could not itself be geocoded."
        ),
    )
    near_source_place: str | None = Field(
        default=None,
        description="Resident-named place resolved by near_source_citation.",
    )
    site_citation: str | None = Field(
        default=None,
        pattern=r"^S\d+$",
        description="Prior FoodHelp DATA citation for the exact site named in a follow-up.",
    )
    urgent: bool = Field(
        default=False,
        description="True when the resident needs food immediately, today, or tonight.",
    )
    service_window: FoodHelpWindow | None = Field(
        default=None,
        description=(
            "Optional resident-requested service window in New York local time. For `tonight`, "
            "pass 17:00-23:59 unless the resident gives a narrower window."
        ),
    )
    service_type: Literal["pantry", "soup_kitchen", "any"] = Field(
        default="any",
        description="Source service type requested by the resident.",
    )


def _food_help_query_schema() -> dict:
    schema = FoodHelpQuery.model_json_schema()
    window = schema.pop("$defs")["FoodHelpWindow"]
    window["description"] = FoodHelpQuery.model_fields["service_window"].description
    schema["properties"]["service_window"] = window
    return schema


class FoodHelpSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_called", "ok", "unavailable"]
    url: str
    fetched_at: datetime | None
    returned_count: int | None
    usable_count: int | None
    complete: bool | None
    requested_limit: int
    next_cursor: str | None = None
    error: Literal["transport_error", "invalid_response"] | None = None


class FoodHelpOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resident_query: str
    label: str
    latitude: float
    longitude: float
    match_type: str | None = None
    provider_id: str | None = None
    confidence: float | None = None
    low_confidence: bool
    precision: Literal["precise", "approximate"]


class FoodHelpOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str


class FoodHelpService(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    service_type: Literal["pantry", "soup_kitchen"]
    eligibility_description: str | None = None
    required_document: list[str] | None = None
    language: list[str] | None = None


class FoodHelpLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    physical_address: str | None = None
    latitude: float
    longitude: float
    service_area: str | None = None
    accessibility: list[str] | None = None


class FoodHelpSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_date: date
    requested_window_start: str | None = None
    requested_window_end: str | None = None
    status: Literal[
        "scheduled_open",
        "scheduled_closed",
        "unknown",
        "conflicting",
    ]
    listed_hours: list[str] | None = None
    source_notes: str | None = None
    overlap_intervals: list[str] | None = None
    holiday_exceptions_available: bool = False
    temporary_exceptions_available: bool = False
    availability_confirmed: bool = False


class FoodHelpServiceAtLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    service_id: str
    location_id: str


class FoodHelpPhone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str
    extension: str | None = None
    type: str | None = None


class FoodHelpRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization: FoodHelpOrganization | None = None
    service: FoodHelpService
    location: FoodHelpLocation
    service_at_location: FoodHelpServiceAtLocation
    schedule: FoodHelpSchedule
    phone: FoodHelpPhone | None = None
    attributes: list[str] | None = None
    distance_miles: float
    distance_method: Literal["haversine"] = "haversine"
    origin_precision: Literal["precise", "approximate"]
    valid_as_of: date | None = None
    citation_id: str
    action_url: str


class FoodHelpImmediateRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phone: Literal["311"] = "311"
    url: str = FOOD_ROUTE_URL
    citation_id: str


FoodHelpOutcome = Literal[
    "success",
    "missing_origin",
    "invalid_site_reference",
    "invalid_service_type",
    "invalid_date",
    "past_date",
    "invalid_service_window",
    "missing_service_window",
    "source_origin_needs_fetch",
    "source_origin_needs_confirmation",
    "location_not_found",
    "location_ambiguous",
    "source_unavailable",
    "no_results",
    "prior_site_missing",
]


class FoodHelpResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: FoodHelpOutcome
    requested_service_type: Literal["pantry", "soup_kitchen", "any"] | None = None
    urgent: bool
    origin: FoodHelpOrigin | None = None
    origin_citation_id: str | None = None
    primary_citation_id: str | None = None
    source_origin_citation_id: str | None = None
    referenced_site_citation_id: str | None = None
    availability_citation_id: str | None = None
    source: FoodHelpSource
    records: list[FoodHelpRecord] = Field(default_factory=list)
    immediate_route: FoodHelpImmediateRoute | None = None
    requested_count: int | None = None
    nearby_checked_count: int | None = None
    citywide_scheduled_open_count: int | None = None
    citywide_unknown_hours_count: int | None = None


def _result(
    outcome: FoodHelpOutcome,
    *,
    service_type: str | None = None,
    urgent: bool = False,
    source_status: Literal["not_called", "ok", "unavailable"] = "not_called",
    fetched_at: datetime | None = None,
    returned_count: int | None = None,
    usable_count: int | None = None,
    source_complete: bool | None = None,
    source_next_cursor: str | None = None,
    source_error: Literal["transport_error", "invalid_response"] | None = None,
    **updates,
) -> FoodHelpResult:
    return FoodHelpResult(
        outcome=outcome,
        requested_service_type=service_type,
        urgent=urgent,
        source=FoodHelpSource(
            status=source_status,
            url=FOODHELP_QUERY_URL,
            fetched_at=fetched_at,
            returned_count=returned_count,
            usable_count=usable_count,
            complete=source_complete,
            requested_limit=FOODHELP_PAGE_SIZE,
            next_cursor=source_next_cursor,
            error=(
                source_error
                or ("transport_error" if source_status == "unavailable" else None)
            ),
        ),
        **updates,
    )


def _origin_result(point: GeoPoint, query: str) -> FoodHelpOrigin:
    return FoodHelpOrigin(
        resident_query=point.resident_query or query,
        label=point.label,
        latitude=point.lat,
        longitude=point.lon,
        match_type=point.match_type or None,
        provider_id=point.provider_id or None,
        confidence=point.confidence,
        low_confidence=point.low_confidence,
        precision=origin_precision(query, point),
    )

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


def _resident_supplied_origin(
    near: str,
    query: str,
    user_turns: tuple[str, ...],
) -> str:
    # F159: took `user_turns`, passed `()`, so only the current message was searchable
    # A confirmation carries no address, so an earlier-turn location was unreachable
    # Widens WHERE we look, not WHAT counts: still resident-authored only
    current_turn = query or (user_turns[-1] if user_turns else "")
    if not current_turn:
        return near
    return resident_supplied_location(
        near, current_turn, user_turns, allow_prior=True
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

_MONTHLY_OCCURRENCE_RE = re.compile(r"\b([1-5])(?:st|nd|rd|th)?\b", re.IGNORECASE)


def _parse_time(value) -> int | None:
    return parse_clock_minutes(value)


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
        if opened is not None and closed is not None and opened < closed:
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
    schedule = {
        weekday: _day_slots(record, prefix, day)
        for weekday, day in enumerate(_DAYS)
    }
    return weekly_open_status(schedule, now)


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
        return bool(_window_overlaps(slots, window_start, window_end))
    if any(_day_slots(record, prefix, day) for day in _DAYS):
        return False
    return None


def _window_overlaps(
    slots: list[tuple[int, int]],
    window_start: int,
    window_end: int,
) -> list[tuple[int, int]]:
    return [
        (max(opened, window_start), min(closed if closed > opened else closed + 1440, window_end))
        for opened, closed in slots
        if opened < window_end
        and (closed if closed > opened else closed + 1440) > window_start
    ]


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


def _listed_hours(record: dict, weekday: int) -> str:
    prefix = _prefix(record)
    day = _DAYS[weekday]
    slots = []
    for n in ("1", "2", "3"):
        opened = _clean(record.get(f"{prefix}_{day}_open{n}"))
        closed = _clean(record.get(f"{prefix}_{day}_close{n}"))
        if (
            opened
            and closed
            and (opened_minutes := _parse_time(opened)) is not None
            and (closed_minutes := _parse_time(closed)) is not None
            and opened_minutes < closed_minutes
        ):
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
    schedule_notes: str
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
        schedule_notes=_clean(record.get("fp_notes")),
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


def _food_route_citation(ctx: ToolContext) -> str:
    return ctx.citations.register(
        FOOD_ROUTE_URL,
        snippet=FOOD_ROUTE_FACT,
        title="Community Food Connection, ACCESS NYC",
        kind="DOC",
        valid_as_of="2026-06-18",
        provenance={"snapshot": {"verified_fact": FOOD_ROUTE_FACT}},
    )


def _availability_citation(
    ctx: ToolContext,
    *,
    pantries: list[FoodPantry],
    nearby: list[FoodPantry],
    now: datetime,
    origin_query: str,
    origin_label: str,
    origin_lat: float,
    origin_lon: float,
    requested: date | None = None,
    service_window: tuple[int, int] | None = None,
    referenced_site: FoodPantry | None = None,
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
        "origin_query": origin_query,
        "origin_label": origin_label,
        "origin_point": [origin_lat, origin_lon],
        "status_filter": WHERE_OPEN,
        "nearby_records_checked": len(nearby),
        "scheduled_open_nearby": scheduled_open_nearby,
    }
    if referenced_site is not None:
        snapshot["referenced_site"] = referenced_site.raw
        snapshot["referenced_site_valid_as_of"] = referenced_site.valid_as_of
        source_url = feature_query_url(
            FOODHELP_URL,
            referenced_site.global_id,
            id_field="GlobalID",
        )
        snippet = f"At lookup, the referenced site had {basis}"
    else:
        snapshot["citywide_records_checked"] = len(pantries)
        snapshot["scheduled_open_citywide"] = scheduled_open_citywide
        source_url = FOODHELP_QUERY_URL
        snippet = (
            f"At lookup, {scheduled_open_nearby} of {len(nearby)} nearby and "
            f"{scheduled_open_citywide} of {len(pantries)} City-listed sites had {basis}"
        )
    return ctx.citations.register(
        source_url,
        snippet=snippet,
        title="NYC FoodHelp availability lookup",
        kind="DATA",
        valid_as_of="",
        provenance=data_provenance(
            snapshot,
            record_id="availability-summary",
            field_pointer="/",
            derivation={"temporal_basis": "weekly_schedule"},
        ),
    )


def _schedule_result(
    pantry: FoodPantry,
    now: datetime,
    requested: date | None = None,
    service_window: tuple[int, int] | None = None,
) -> FoodHelpSchedule:
    requested_day = requested or now.date()
    show_hours = _monthly_occurrence_allows(
        pantry.raw, requested_day
    ) is not False
    if service_window:
        start, end = service_window
        scheduled = _scheduled_during(pantry.raw, requested_day, start, end)
        overlaps = _window_overlaps(
            _day_slots(
                pantry.raw,
                _prefix(pantry.raw),
                _DAYS[requested_day.weekday()],
            ),
            start,
            end,
        )
        overlap_intervals = [
            f"{opened // 60:02d}:{opened % 60:02d}-{closed // 60:02d}:{closed % 60:02d}"
            for opened, closed in overlaps
        ]
    elif requested:
        overlap_intervals = None
        if _schedule_conflict(pantry.raw, requested_day):
            scheduled = "conflicting"
        elif _monthly_occurrence_allows(pantry.raw, requested_day) is False:
            scheduled = False
        else:
            scheduled = _scheduled_on(pantry.raw, requested_day)
    else:
        overlap_intervals = None
        scheduled = _open_now(pantry.raw, now)
    status = (
        "conflicting"
        if scheduled == "conflicting"
        else "scheduled_open"
        if scheduled is True
        else "scheduled_closed"
        if scheduled is False
        else "unknown"
    )
    hours = _listed_hours(pantry.raw, requested_day.weekday())
    return FoodHelpSchedule(
        requested_date=requested_day,
        requested_window_start=(
            f"{service_window[0] // 60:02d}:{service_window[0] % 60:02d}"
            if service_window
            else None
        ),
        requested_window_end=(
            f"{service_window[1] // 60:02d}:{service_window[1] % 60:02d}"
            if service_window
            else None
        ),
        status=status,
        listed_hours=hours.split(", ") if hours and show_hours else None,
        source_notes=pantry.schedule_notes or None,
        overlap_intervals=overlap_intervals or None,
    )


def _record_result(
    ctx: ToolContext,
    pantry: FoodPantry,
    *,
    origin: GeoPoint,
    origin_query: str,
    now: datetime,
    requested: date | None,
    service_window: tuple[int, int] | None,
) -> FoodHelpRecord:
    dist_mi = miles(haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon))
    citation_id = _pantry_citation(
        ctx,
        pantry,
        origin_lat=origin.lat,
        origin_lon=origin.lon,
        dist_mi=dist_mi,
    )
    service_type = "soup_kitchen" if _prefix(pantry.raw) == "sk" else "pantry"
    flags = _flags(pantry)
    service_id = f"{pantry.global_id}:service"
    location_id = f"{pantry.global_id}:location"
    return FoodHelpRecord(
        service=FoodHelpService(
            id=service_id,
            name=pantry.name,
            service_type=service_type,
        ),
        location=FoodHelpLocation(
            id=location_id,
            name=pantry.name,
            physical_address=pantry.address or None,
            latitude=pantry.lat,
            longitude=pantry.lon,
        ),
        service_at_location=FoodHelpServiceAtLocation(
            id=pantry.global_id,
            service_id=service_id,
            location_id=location_id,
        ),
        schedule=_schedule_result(pantry, now, requested, service_window),
        phone=FoodHelpPhone(number=pantry.phone) if pantry.phone else None,
        attributes=flags or None,
        distance_miles=dist_mi,
        origin_precision=origin_precision(origin_query, origin),
        valid_as_of=date.fromisoformat(pantry.valid_as_of) if pantry.valid_as_of else None,
        citation_id=citation_id,
        action_url=directions_link(pantry.lat, pantry.lon),
    )


async def _handler(args: dict, ctx: ToolContext) -> FoodHelpResult:
    near = (args.get("near") or "").strip()
    if not near:
        return _result("missing_origin", urgent=args.get("urgent") is True)
    site_citation = str(args.get("site_citation") or "").strip()
    site_record_id = ""
    if site_citation:
        citation = ctx.citations.mapping().get(site_citation, {})
        provenance = citation.get("provenance") or {}
        if (
            citation.get("kind") != "DATA"
            or not str(citation.get("url") or "").startswith(FOODHELP_URL)
            or not provenance.get("record_id")
        ):
            return _result(
                "invalid_site_reference",
                urgent=args.get("urgent") is True,
                referenced_site_citation_id=site_citation,
            )
        site_record_id = str(provenance["record_id"])
    service_type = str(args.get("service_type") or "any")
    if service_type not in {"pantry", "soup_kitchen", "any"}:
        return _result("invalid_service_type", urgent=args.get("urgent") is True)
    now = datetime.now(_NYC_TZ)
    visit_date = args.get("visit_date")
    try:
        requested = (
            visit_date
            if isinstance(visit_date, date)
            else date.fromisoformat(str(visit_date))
            if visit_date
            else None
        )
    except (TypeError, ValueError):
        return _result(
            "invalid_date",
            service_type=service_type,
            urgent=args.get("urgent") is True,
        )
    if requested and requested < now.date():
        return _result(
            "past_date",
            service_type=service_type,
            urgent=args.get("urgent") is True,
        )
    window = args.get("service_window") or {}
    window_start = str(window.get("start") or "").strip()
    window_end = str(window.get("end") or "").strip()
    service_window = None
    if window_start:
        start = _parse_time(window_start)
        end = _parse_time(window_end)
        if start is None or end is None or end <= start:
            return _result(
                "invalid_service_window",
                service_type=service_type,
                urgent=args.get("urgent") is True,
            )
        service_window = (start, end)
    elif args.get("visit_time"):
        visit_time = args["visit_time"]
        try:
            requested_time = (
                visit_time
                if isinstance(visit_time, time)
                else time.fromisoformat(str(visit_time))
            )
        except (TypeError, ValueError):
            return _result(
                "invalid_service_window",
                service_type=service_type,
                urgent=args.get("urgent") is True,
            )
        minute = requested_time.hour * 60 + requested_time.minute
        service_window = (minute, minute + 1)
    future_schedule = requested is not None and requested > now.date()
    requested_day = requested or now.date()
    urgent = args.get("urgent") is True and not future_schedule
    if args.get("urgent") is True and not future_schedule and service_window is None:
        return _result(
            "missing_service_window",
            service_type=service_type,
            urgent=True,
        )
    source_id = str(args.get("near_source_citation") or "")
    source_place = str(args.get("near_source_place") or "")
    stored_origin = current_resolved_location(near, ctx)
    resident_origin = _resident_supplied_origin(near, ctx.query, ctx.user_turns)
    resident_origin = resident_origin or (
        stored_origin.resident_query if stored_origin else ""
    )
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
            return _result(
                "source_origin_needs_confirmation",
                service_type=service_type,
                urgent=urgent,
                source_origin_citation_id=source_id,
            )
        return _result(
            "source_origin_needs_fetch",
            service_type=service_type,
            urgent=urgent,
            source_origin_citation_id=source_id,
        )
    resident_origin = resident_origin or source_origin
    if not resident_origin:
        return _result("missing_origin", service_type=service_type, urgent=urgent)
    near = resident_origin

    if stored_origin is None:
        ctx.current_location = None
    origin = await resolve_location(near, ctx)
    if origin is None:
        return _result(
            "location_not_found",
            service_type=service_type,
            urgent=urgent,
            source_origin_citation_id=source_id or None,
        )
    if origin.low_confidence:
        return _result(
            "location_ambiguous",
            service_type=service_type,
            urgent=urgent,
            origin=_origin_result(origin, near),
            source_origin_citation_id=source_id or None,
        )
    if origin.resident_query:
        ctx.current_location = origin

    query_where = WHERE_OPEN
    if site_record_id:
        escaped_site_id = site_record_id.replace("'", "''")
        query_where = f"{WHERE_OPEN} AND GlobalID='{escaped_site_id}'"
    try:
        page = await query_feature_service_page(
            FOODHELP_URL,
            where=query_where,
            result_record_count=FOODHELP_PAGE_SIZE,
            client=ctx.http,
        )
    except httpx.HTTPError:
        return _result(
            "source_unavailable",
            service_type=service_type,
            urgent=urgent,
            source_status="unavailable",
            fetched_at=now,
            origin=_origin_result(origin, near),
        )
    except (ValueError, TypeError, AttributeError):
        return _result(
            "source_unavailable",
            service_type=service_type,
            urgent=urgent,
            source_status="unavailable",
            source_error="invalid_response",
            fetched_at=now,
            origin=_origin_result(origin, near),
        )

    records = page.records
    source_returned_count = len(records)
    source_next_cursor = (
        page.pagination_token
        or (f"offset:{page.next_offset}" if page.next_offset is not None else None)
    )
    pantries = [p for p in (_to_pantry(r) for r in records) if p is not None]
    if service_type != "any":
        expected_prefix = "fp" if service_type == "pantry" else "sk"
        pantries = [pantry for pantry in pantries if _prefix(pantry.raw) == expected_prefix]
    if not pantries:
        cite = _availability_citation(
            ctx,
            pantries=[],
            nearby=[],
            now=now,
            origin_query=near,
            origin_label=origin.label,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            requested=requested_day,
            service_window=service_window,
        )
        if urgent:
            ctx.response_priority_citation_ids.add(cite)
            route_cite = _food_route_citation(ctx)
            ctx.response_priority_citation_ids.add(route_cite)
            immediate_route = FoodHelpImmediateRoute(citation_id=route_cite)
        else:
            immediate_route = None
        return _result(
            "no_results",
            service_type=service_type,
            urgent=urgent,
            source_status="ok",
            fetched_at=now,
            returned_count=source_returned_count,
            usable_count=0,
            source_complete=page.complete,
            source_next_cursor=source_next_cursor,
            origin=_origin_result(origin, near),
            origin_citation_id=resolved_location_citation(ctx, origin),
            source_origin_citation_id=source_id or None,
            availability_citation_id=cite,
            immediate_route=immediate_route,
            nearby_checked_count=0,
            citywide_scheduled_open_count=0,
        )
    k = int(args.get("max_results") or 5)
    unique = [
        pantry
        for pantry, _distance_m in rank_nearby(
            origin,
            pantries,
            key=lambda pantry: (
                pantry.name.strip().casefold(),
                round(pantry.lat, 5),
                round(pantry.lon, 5),
            ),
        )
    ]
    if site_record_id:
        referenced = [
            pantry
            for pantry in unique
            if pantry.global_id.casefold() == site_record_id.casefold()
        ]
        if not referenced:
            return _result(
                "prior_site_missing",
                service_type=service_type,
                urgent=urgent,
                source_status="ok",
                fetched_at=now,
                returned_count=source_returned_count,
                usable_count=len(unique),
                source_complete=page.complete,
                source_next_cursor=source_next_cursor,
                origin=_origin_result(origin, near),
                origin_citation_id=resolved_location_citation(ctx, origin),
                referenced_site_citation_id=site_citation,
            )
        rank_pool = referenced
    else:
        rank_pool = unique
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

        rank_pool.sort(key=requested_window_rank)
    elif future_schedule:
        def requested_day_rank(pantry: FoodPantry) -> tuple[float, float]:
            scheduled = _scheduled_on(pantry.raw, requested)
            return (
                0 if scheduled is True else 2 if scheduled is False else 1,
                haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon),
            )

        rank_pool.sort(key=requested_day_rank)
    elif not urgent and not site_record_id:
        def current_usability_rank(pantry: FoodPantry) -> tuple[float, float]:
            scheduled = _open_now(pantry.raw, now)
            return (
                0 if scheduled is True else 2 if scheduled is False else 1,
                haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon),
            )

        rank_pool.sort(key=current_usability_rank)
    ranked = rank_pool[:k]
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
    citywide_scheduled_open = len(citywide_open)
    citywide_unknown_hours = sum(availability(pantry) is None for pantry in unique)
    availability_citation = (
        _availability_citation(
            ctx,
            pantries=unique,
            nearby=ranked,
            now=now,
            origin_query=near,
            origin_label=origin.label,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            requested=requested_day,
            service_window=service_window,
            referenced_site=ranked[0] if site_record_id else None,
        )
        if urgent
        else ""
    )
    if availability_citation:
        ctx.response_priority_citation_ids.add(availability_citation)
    route_citation = _food_route_citation(ctx) if urgent else ""
    if route_citation:
        ctx.response_priority_citation_ids.add(route_citation)
    displayed = (
        ranked[:1]
        if site_record_id
        else scheduled_open[:1]
        if urgent and service_window
        else scheduled_open or citywide_open[:1] or unknown_hours
        if urgent
        else ranked
    )
    origin_citation = resolved_location_citation(ctx, origin)
    displayed_records = [
        _record_result(
            ctx,
            pantry,
            origin=origin,
            origin_query=near,
            now=now,
            requested=requested,
            service_window=service_window,
        )
        for pantry in displayed
    ]
    return _result(
        "success",
        service_type=service_type,
        urgent=urgent,
        source_status="ok",
        fetched_at=now,
        returned_count=source_returned_count,
        usable_count=len(unique),
        source_complete=page.complete,
        source_next_cursor=source_next_cursor,
        origin=_origin_result(origin, near),
        origin_citation_id=origin_citation,
        primary_citation_id=(
            displayed_records[0].citation_id if displayed_records else None
        ),
        source_origin_citation_id=source_id or None,
        referenced_site_citation_id=site_citation or None,
        availability_citation_id=availability_citation or None,
        records=displayed_records,
        immediate_route=(
            FoodHelpImmediateRoute(citation_id=route_citation)
            if route_citation
            else None
        ),
        requested_count=k,
        nearby_checked_count=len(ranked),
        citywide_scheduled_open_count=(
            None if site_record_id else citywide_scheduled_open
        ),
        citywide_unknown_hours_count=(None if site_record_id else citywide_unknown_hours),
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_foodhelp_locations",
            description=(
                "Find the nearest City-listed NYC food pantries / soup kitchens to an address, grounded in "
                "the city's official FoodHelp data (finder.nyc.gov/foodhelp). Pass `near` = the "
                "user's NYC address or neighborhood. If they have not supplied one, ask before "
                "calling and never invent a broad origin. Pass `service_type=pantry` for a pantry "
                "request, `soup_kitchen` for a soup-kitchen request, or `any` for general food "
                "help. Pass `visit_date` for a requested date and `visit_time` for a specific "
                "New York local time; `max_results` defaults to 5. Returns "
                "each site's name, full address, requested-date weekly hours or current-day "
                "scheduled-open status, phone, dietary/access type "
                "(Halal/Kosher/HIV/Mobile), and a Google Maps directions link, every site cited. "
                "Set `urgent=true` when the resident needs food now, today, or tonight so the "
                "result leads with the immediate fallback and does not overstate weekly hours. "
                "Whenever `urgent=true`, also pass `service_window` with start and end as 24-hour "
                "NYC-local HH:MM values: now is the current NYC minute, today runs from the current "
                "NYC time through 23:59, and tonight is 17:00-23:59 unless the resident gives a "
                "narrower window. "
                "When a follow-up refers to one site returned earlier, pass that site's citation "
                "ID as `site_citation` so the lookup does not silently switch locations. "
                "NEVER guess a pantry: if geocoding fails or none are near, say so and point to 311. "
                "The source has no language or eligibility data, so do not infer either."
            ),
            parameters=_food_help_query_schema(),
            handler=_handler,
            open_world=True,  # hits the live ArcGIS FoodHelp service + geocoder
            return_type=FoodHelpResult,
        )
    ]
