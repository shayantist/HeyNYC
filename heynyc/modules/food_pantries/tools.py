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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    _clarify_message,
    _resolution_note,
    geocode,
    haversine_m,
    miles,
)

# The live backend of finder.nyc.gov/foodhelp, verified public + tokenless.
FOODHELP_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Food_Help_Programs_PROD_view/FeatureServer/0"
)
WHERE_OPEN = "status='Open'"
OFFICIAL = "finder.nyc.gov/foodhelp or call 311"
NO_LOCATION = (
    "The proposed search origin was not supplied by the user, so do not use it. "
    "For immediate food help, tell the user to call 311 or use "
    "https://finder.nyc.gov/foodhelp. Also ask where they are, using an NYC address or "
    "neighborhood, so the next search can return nearby options. Never guess their location."
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
_ORIGIN_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ORIGIN_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|don['’]?t|do\s+not)\b", re.IGNORECASE)
_ORIGIN_STALE_RE = re.compile(r"\b(?:used to|formerly|previously)\b", re.IGNORECASE)
_ORIGIN_TOKEN_ALIASES = {
    "st": "street", "ave": "avenue", "rd": "road", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "pkwy": "parkway",
}
def _clean(value) -> str:
    """None / literal 'NULL' / blanks → ''. ArcGIS returns JSON null for empty fields."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() == "NULL" else text


def _resident_supplied_origin(near: str, query: str, user_turns: tuple[str, ...]) -> str:
    current_turn = query or (user_turns[-1] if user_turns else "")
    if not current_turn:
        return near
    parts = [part.strip() for part in near.split(",") if part.strip()]
    candidates = [", ".join(parts[:end]) for end in range(len(parts), 0, -1)]
    turn_matches = list(_ORIGIN_TOKEN_RE.finditer(current_turn))
    turn_tokens = [_ORIGIN_TOKEN_ALIASES.get(match.group().lower(), match.group().lower())
                   for match in turn_matches]
    for candidate in candidates:
        tokens = [_ORIGIN_TOKEN_ALIASES.get(token.lower(), token.lower())
                  for token in _ORIGIN_TOKEN_RE.findall(candidate)]
        if len(tokens) < 2 and len(parts) > 1:
            continue
        for start in range(len(turn_tokens) - len(tokens) + 1):
            if tokens != turn_tokens[start:start + len(tokens)]:
                continue
            raw_start = turn_matches[start].start()
            raw_end = turn_matches[start + len(tokens) - 1].end()
            clause_start = max(current_turn.rfind(mark, 0, raw_start) for mark in ".!?;,") + 1
            clause_ends = [current_turn.find(mark, raw_end) for mark in ".!?;,"]
            clause_end = min((end for end in clause_ends if end >= 0), default=len(current_turn))
            # ponytail: lexical negation is a fail-closed floor; semantic review covers other languages.
            clause = current_turn[clause_start:clause_end]
            if _ORIGIN_NEGATION_RE.search(clause) or _ORIGIN_STALE_RE.search(clause):
                continue
            return current_turn[raw_start:raw_end]
    return ""


# --- hours / open-now ------------------------------------------------------

_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$")


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
    return "sk" if _clean(record.get("program_type")).upper() == "SK" else "fp"


def _day_slots(record: dict, prefix: str, day: str) -> list[tuple[int, int]]:
    """The (open, close) minute-ranges for one weekday (up to three windows)."""
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


def _status_label(open_now: bool | None) -> str:
    if open_now is True:
        return "scheduled open now"
    if open_now is False:
        return "scheduled closed now"
    return "hours not listed, call ahead"


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
                    "distance_mi": dist_mi},
    )
    return ctx.citations.register(
        url,
        snippet=f"{pantry.name}, {pantry.address}",
        title="NYC FoodHelp (Food Help Programs)",
        kind="DATA",
        valid_as_of=pantry.valid_as_of,
        provenance=provenance,
    )


def _pantry_block(pantry: FoodPantry, cite: str, dist_mi: float, now: datetime) -> str:
    flags = _flags(pantry)
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    status = _status_label(_open_now(pantry.raw, now))
    parts = [f"- {pantry.name}{flag_str} ({pantry.address or 'NYC'}), "
             f"{dist_mi:.2f} mi straight-line, {status} {{cite:{cite}}}"]
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
    resident_origin = _resident_supplied_origin(near, ctx.query, ctx.user_turns)
    if not resident_origin:
        return NO_LOCATION
    near = resident_origin

    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return (f"I couldn't locate '{near}' in NYC, so I can't find a nearby food pantry. Ask the "
                f"user for a specific NYC address or neighborhood, don't guess a pantry.")
    if origin.low_confidence:
        return _clarify_message(near)

    try:
        records = await query_feature_service(FOODHELP_URL, where=WHERE_OPEN, client=ctx.http)
    except httpx.HTTPError:
        return (f"I couldn't reach the city's FoodHelp data right now, don't guess a pantry. "
                f"Point the user to {OFFICIAL}.")

    pantries = [p for p in (_to_pantry(r) for r in records) if p is not None]
    if not pantries:
        return (f"No open food pantries came back from the city's FoodHelp data. Don't invent one, "
                f"point the user to {OFFICIAL}.")

    k = int(args.get("k") or 5)
    now = datetime.now(_NYC_TZ)
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
    ranked = unique[:k]

    lines = [
        f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})",
        _resolution_note(near, origin),
        "Nearest City-listed food pantry candidates from NYC FoodHelp "
        "(finder.nyc.gov/foodhelp), report only these, cite each:",
    ]
    scheduled_open_count = sum(_open_now(pantry.raw, now) is True for pantry in ranked)
    citywide_scheduled_open = sum(_open_now(pantry.raw, now) is True for pantry in unique)
    citywide_unknown_hours = sum(_open_now(pantry.raw, now) is None for pantry in unique)
    if scheduled_open_count:
        verb = "is" if scheduled_open_count == 1 else "are"
        lines.append(f"{scheduled_open_count} of these candidates {verb} scheduled open now.")
    else:
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
    for pantry in ranked:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, pantry.lat, pantry.lon))
        cite = _pantry_citation(ctx, pantry, origin_lat=origin.lat, origin_lon=origin.lon,
                                dist_mi=dist_mi)
        lines.append(_pantry_block(pantry, cite, dist_mi, now))
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
                "user's NYC address or neighborhood; optional `k` (default 5). Returns each site's "
                "name, full address, scheduled-open-now status, phone, dietary/access type "
                "(Halal/Kosher/HIV/Mobile), and a Google Maps directions link, every site cited. "
                "NEVER guess a pantry: if geocoding fails or none are near, say so and point to 311. "
                "The source has no language info and eligibility notes are often blank, don't invent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string",
                             "description": "The NYC address or neighborhood to search near."},
                    "k": {"type": "integer",
                          "description": "How many pantries to return (default 5).", "default": 5},
                },
                "required": ["near"],
            },
            handler=_handler,
            open_world=True,  # hits the live ArcGIS FoodHelp service + geocoder
        )
    ]
