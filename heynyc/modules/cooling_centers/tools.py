"""Live lookup for NYC cooling centers and other Cool Options."""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import ResidentFact, Tool, ToolContext
from heynyc.core.tools.geo import (
    _requested_result_limit,
    _resolution_note,
    geocode,
    haversine_m,
    maps_link,
    miles,
    resident_supplied_location,
)

COOL_OPTIONS_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Cool_Options/FeatureServer/0"
)
_NYC_TZ = ZoneInfo("America/New_York")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_KINDS = ("all", "indoor", "cooling_center")
_AUDIENCES = ("any", "not_age_restricted")
_SELECTED_SITE_FACT = "/cooling/site"
_OFFERED_SITE_FACT = "/cooling/offered"


def _nyc_now() -> datetime:
    return datetime.now(_NYC_TZ)


def _minutes(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, "%I:%M %p")
    except ValueError:
        return None
    return parsed.hour * 60 + parsed.minute


def _open_now(record: dict, now: datetime) -> bool | None:
    day = _DAYS[now.weekday()]
    previous_day = _DAYS[(now.weekday() - 1) % 7]
    current = now.hour * 60 + now.minute
    known = False
    for interval in (1, 2):
        opened = _minutes(record.get(f"cc_{day}_open{interval}"))
        closed = _minutes(record.get(f"cc_{day}_close{interval}"))
        if opened is None or closed is None:
            continue
        known = True
        if (closed > opened and opened <= current < closed) or (
            closed <= opened and current >= opened
        ):
            return True
    for interval in (1, 2):
        opened = _minutes(record.get(f"cc_{previous_day}_open{interval}"))
        closed = _minutes(record.get(f"cc_{previous_day}_close{interval}"))
        if opened is None or closed is None or closed > opened:
            continue
        known = True
        if current < closed:
            return True
    return False if known else None


def _next_open(record: dict, now: datetime) -> tuple[int, int, str] | None:
    """Earliest scheduled reopening from `now`: (days_ahead, open_minute, 'Day HH:MM AM').

    Scans the row's own cc_<day>_open<interval> fields forward up to a week; skips
    intervals that already opened earlier today. Returns None if none are known.
    """
    current = now.hour * 60 + now.minute
    best: tuple[int, int, str] | None = None
    for offset in range(7):
        weekday = (now.weekday() + offset) % 7
        day = _DAYS[weekday]
        for interval in (1, 2):
            opened = _minutes(record.get(f"cc_{day}_open{interval}"))
            if opened is None or (offset == 0 and opened <= current):
                continue
            time_text = str(record.get(f"cc_{day}_open{interval}")).strip()
            candidate = (offset, opened, f"{_DAY_NAMES[weekday]} {time_text}")
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    return best


def _scheduled_hours(record: dict, weekday: int) -> str:
    displayed = str(record.get(_DAY_NAMES[weekday]) or "").strip()
    if displayed:
        return displayed
    day = _DAYS[weekday]
    intervals = []
    for interval in (1, 2):
        opened = str(record.get(f"cc_{day}_open{interval}") or "").strip()
        closed = str(record.get(f"cc_{day}_close{interval}") or "").strip()
        if opened and closed:
            intervals.append(f"{opened}-{closed}")
    return ", ".join(intervals)


def _scheduled_open(record: dict, weekday: int) -> bool | None:
    displayed = str(record.get(_DAY_NAMES[weekday]) or "").strip()
    if displayed.casefold().startswith("closed"):
        return False
    day = _DAYS[weekday]
    for interval in (1, 2):
        if (
            _minutes(record.get(f"cc_{day}_open{interval}")) is not None
            and _minutes(record.get(f"cc_{day}_close{interval}")) is not None
        ):
            return True
    return None


def _value(record: dict, mixed: str, upper: str) -> str:
    return str(record.get(mixed) or record.get(upper) or "").strip()


def _edit_date(record: dict) -> str:
    value = record.get("EditDate")
    if not isinstance(value, (int, float)):
        return ""
    return datetime.fromtimestamp(value / 1000, tz=UTC).date().isoformat()


def _citation(ctx: ToolContext, item: dict) -> str:
    record = item["record"]
    record_id = str(record.get("OBJECTID", ""))
    url = feature_query_url(item["source"], record_id) if record_id else item["source"]
    return ctx.citations.register(
        url,
        snippet=f"{item['name']}, {item['type']}, status: {record.get('Finder_status', '')}",
        title="NYC Emergency Management Cool Options",
        kind="DATA",
        valid_as_of=_edit_date(record),
        provenance=data_provenance(record, record_id=record_id, field_pointer="/"),
    )


def _item(record: dict) -> dict | None:
    try:
        lat, lon = float(record["lat"]), float(record["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    item_type = str(record.get("Space_type") or "cool option").strip().lower()
    active = item_type == "cooling center"
    if active:
        item_type = "activated cooling center"
    # F072: preserve the finder row's audience restriction so an older-adults-only center is never
    # handed to a parent as if it were all-ages.
    age_restriction = str(record.get("Age_restriction") or "").strip()
    audience = "Age-restricted" if age_restriction.lower() == "yes" else ""
    age_restricted = age_restriction.lower() == "yes"
    return {
        "record": record,
        "source": COOL_OPTIONS_URL,
        "name": _value(record, "Facility_name", "FACILITY_NAME"),
        "address": _value(record, "Address", "ADDRESS"),
        "phone": _value(record, "Phone", "PHONE"),
        "accessible": str(record.get("Accessible") or "").strip(),
        "pet_friendly": _value(record, "Pet_friendly", "PET_FRIENDLY"),
        "type": item_type,
        "audience": audience,
        "age_restricted": age_restricted,
        "not_age_restricted": age_restriction.lower() == "no",
        "lat": lat,
        "lon": lon,
        "active": active,
    }


def _site_key(item: dict) -> str:
    record = item["record"]
    return str(record.get("NYCEM_ID") or f"{item['name']}|{item['address']}").casefold()


def _site_tokens(value: object) -> list[str]:
    return re.findall(r"[^\W_]+", str(value or "").casefold())


def _contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    return any(
        haystack[start:start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _match_site(
    items: list[dict], turn: str, requested: str = "", near: str = ""
) -> tuple[dict | None, set[str]]:
    turn_tokens = _site_tokens(turn)
    requested_tokens = _site_tokens(requested)
    near_tokens = _site_tokens(near)
    matches: list[dict] = []
    excluded: set[str] = set()

    for item in items:
        name_tokens = _site_tokens(item.get("name"))
        if not name_tokens:
            continue
        exact = _contains_tokens(turn_tokens, name_tokens) and (
            not requested_tokens or requested_tokens == name_tokens
        )
        proof = name_tokens if exact else requested_tokens or name_tokens[:1]
        if requested_tokens and not _contains_tokens(turn_tokens, proof):
            proof = requested_tokens[:1]
        if (
            (not exact and requested_tokens and name_tokens[:len(requested_tokens)] != requested_tokens)
            or not _contains_tokens(turn_tokens, proof)
            or _contains_tokens(near_tokens, proof)
        ):
            continue
        if resident_supplied_location(" ".join(proof), turn, (turn,)):
            matches.append(item)
        else:
            excluded.add(_site_key(item))

    return (matches[0] if len(matches) == 1 else None), excluded


def _site_from_turn(items: list[dict], turn: str, requested: str = "") -> dict | None:
    return _match_site(items, turn, requested)[0]


def _decode_site_fact(
    value: object, *, offered: bool = False
) -> tuple[list[str], list[float], dict[str, str] | None] | None:
    if not isinstance(value, dict):
        return None
    raw_keys = value.get("keys") if offered else [value.get("key")]
    origin = value.get("origin")
    scope = value.get("scope") if offered else None
    if (
        not isinstance(raw_keys, list)
        or not raw_keys
        or any(not isinstance(key, str) or not key.strip() for key in raw_keys)
        or not isinstance(origin, list)
        or len(origin) != 2
        or any(isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) for coordinate in origin)
        or offered
        and (
            not isinstance(scope, dict)
            or set(scope) != {"kind", "audience"}
            or any(not isinstance(value, str) or not value for value in scope.values())
            or scope.get("kind") not in _KINDS
            or scope.get("audience") not in _AUDIENCES
        )
    ):
        return None
    return [key.casefold() for key in raw_keys], [float(origin[0]), float(origin[1])], scope


async def _cool_options_lookup(args: dict, ctx: ToolContext) -> str:
    near = str(args.get("near", "")).strip()
    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
    if origin.low_confidence:
        return f"'{near}' may match several places. Ask for a specific NYC address or landmark."

    try:
        records = await query_feature_service(
            COOL_OPTIONS_URL,
            where="Finder_status='OPEN'",
            result_record_count=2000,
            client=ctx.http,
        )
    except Exception:
        return "The NYC Cool Options finder was unavailable, so I cannot safely list locations."

    kind = str(args.get("kind", "all")).strip().casefold()
    audience = str(args.get("audience", "any")).strip().casefold()
    items = []
    for record in records:
        space_type = str(record.get("Space_type") or "").strip().lower()
        if kind == "cooling_center" and space_type != "cooling center":
            continue
        if kind == "indoor" and str(record.get("Location_type") or "").strip().lower() != "indoor":
            continue
        item = _item(record)
        if item and (
            audience != "not_age_restricted" or item["not_age_restricted"]
        ):
            items.append(item)

    seen: set[str] = set()
    unique = []
    for item in items:
        record_id = str(item["record"].get("NYCEM_ID") or "").lower()
        key = record_id or f"{item['name'].lower()}|{item['address'].lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    if not unique:
        if kind == "cooling_center":
            return (
                "No activated cooling centers were returned by the NYC finder. "
                "Try kind='indoor' for other indoor Cool Options."
            )
        return "No matching NYC Cool Options were found near that location."

    turn = ctx.user_turns[-1] if ctx.user_turns else ""
    origin_value = [origin.lat, origin.lon]
    offered_fact = ctx.resident_facts.get(_OFFERED_SITE_FACT)
    offered_state = (
        _decode_site_fact(offered_fact.value, offered=True)
        if offered_fact
        else None
    )
    scope = {"kind": kind, "audience": audience}
    if offered_state is None or offered_state[2] != scope:
        ctx.resident_facts.pop(_OFFERED_SITE_FACT, None)
        ctx.resident_facts.pop(_SELECTED_SITE_FACT, None)
        offered_state = None
    elif offered_state[1] != origin_value:
        ctx.resident_facts.pop(_OFFERED_SITE_FACT, None)
        ctx.resident_facts.pop(_SELECTED_SITE_FACT, None)
        offered_state = None

    if offered_state is None:
        offered_keys = [_site_key(item) for item in unique]
        ctx.resident_facts[_OFFERED_SITE_FACT] = ResidentFact(
            value={"keys": offered_keys, "origin": origin_value, "scope": scope},
            source_turn_id=str(len(ctx.user_turns)),
            status="captured",
        )
    else:
        offered_keys = offered_state[0]
    offered_items = [item for item in unique if _site_key(item) in offered_keys]
    selected, excluded = _match_site(
        offered_items,
        turn,
        str(args.get("site") or "").strip(),
        near,
    )
    if selected:
        unique = [selected]
        ctx.resident_facts[_SELECTED_SITE_FACT] = ResidentFact(
            value={"key": _site_key(selected), "origin": origin_value},
            source_turn_id=str(len(ctx.user_turns)),
            status="captured",
        )
    else:
        selected_fact = ctx.resident_facts.get(_SELECTED_SITE_FACT)
        selected_state = _decode_site_fact(selected_fact.value) if selected_fact else None
        if selected_state and selected_state[1] == origin_value and selected_state[0][0] not in excluded:
            stored_item = next(
                (item for item in unique if _site_key(item) == selected_state[0][0]),
                None,
            )
            if stored_item and _site_key(stored_item) in offered_keys:
                unique = [stored_item]
            else:
                return "I couldn't re-confirm the cooling site you selected in the current City finder."
        elif selected_fact:
            ctx.resident_facts.pop(_SELECTED_SITE_FACT, None)
    if excluded:
        unique = [item for item in unique if _site_key(item) not in excluded]

    now = _nyc_now()
    requested_on = str(args.get("on") or "").strip()
    try:
        target_date = date.fromisoformat(requested_on) if requested_on else now.date()
    except ValueError:
        return "The `on` date must use YYYY-MM-DD. Ask the resident to clarify the date."
    planning_ahead = target_date != now.date()
    target_day_name = _DAY_NAMES[target_date.weekday()]
    for item in unique:
        item["open_now"] = _open_now(item["record"], now)
        item["target_hours"] = _scheduled_hours(item["record"], target_date.weekday())
        item["target_open"] = _scheduled_open(item["record"], target_date.weekday())
        item["distance_m"] = haversine_m(origin.lat, origin.lon, item["lat"], item["lon"])
    if planning_ahead:
        unique.sort(
            key=lambda item: (
                0 if item["target_open"] is True else 2 if item["target_open"] is False else 1,
                item["distance_m"],
            )
        )
    else:
        unique.sort(
            key=lambda item: (
                0 if item["open_now"] is True else 2 if item["open_now"] is False else 1,
                item["distance_m"],
            )
        )

    # F068: when the nearest open site is farther than sites that are closed right
    # now, say so in data terms so the answer is framed honestly instead of reading
    # "nearest option is a pet store 2 miles away" with no context.
    nearest_open = (
        None
        if planning_ahead
        else next((item for item in unique if item["open_now"] is True), None)
    )
    closer_closed = (
        [
            item
            for item in unique
            if item["open_now"] is False and item["distance_m"] < nearest_open["distance_m"]
        ]
        if nearest_open
        else []
    )

    limit = _requested_result_limit(args.get("limit", 3), ctx.query)
    selected = unique[:limit]
    day_name = _DAY_NAMES[now.weekday()]
    target_date_label = (
        f"{target_date.strftime('%A, %B')} {target_date.day}, {target_date.year}"
    )
    date_scope = f" for {target_date_label}" if planning_ahead else ""
    lines = [
        f"NYC Cool Options{date_scope} near {origin.label}:",
        _resolution_note(near, origin),
    ]
    if planning_ahead:
        lines.append(
            "Activation status is current at lookup time, not a guarantee for the requested date."
        )
    if closer_closed:
        count = len(closer_closed)
        note = (
            f"{count} closer {'option is' if count == 1 else 'options are'} "
            "scheduled closed right now"
        )
        reopenings = [r for item in closer_closed if (r := _next_open(item["record"], now))]
        if reopenings:
            note += f"; the soonest reopens {min(reopenings)[2]}"
        lines.append(note + ".")
    for index, item in enumerate(selected, 1):
        cite = _citation(ctx, item)
        distance = miles(item["distance_m"])
        item_type = (
            "cooling center"
            if planning_ahead and item["active"]
            else item["type"]
        )
        lines.append(
            f"{index}. {item['name']}, {item_type}, {distance:.2f} miles {{cite:{cite}}}"
        )
        if planning_ahead and item["active"]:
            lines.append(
                f"   Activation: current at lookup only; not verified for {target_date_label}"
            )
        if item["age_restricted"]:
            # F072: prominent, language-independent restriction data (the row's own audience);
            # the model translates "Older Adult Center" / "age-restricted" naturally.
            label = item["audience"] or "Age-restricted"
            lines.append(f"   Audience: {label} (age-restricted, not open to all ages)")
        elif audience == "not_age_restricted":
            lines.append(f"   Audience: City row is not marked age-restricted {{cite:{cite}}}")
        if item["address"]:
            lines.append(f"   {item['address']}")
        if planning_ahead and item["target_hours"]:
            lines.append(
                f"   Scheduled {target_date_label}: {item['target_hours']}"
            )
        elif planning_ahead:
            lines.append(f"   No {target_day_name} hours are listed in this City record")
        else:
            hours = str(item["record"].get(day_name) or "").strip()
            if item["open_now"] is True:
                status = "scheduled open now"
            elif item["open_now"] is False:
                status = "scheduled closed now"
            else:
                status = "current schedule unclear"
            lines.append(f"   {status}" + (f". {day_name}: {hours}" if hours else ""))
        flags = []
        if item["accessible"]:
            flags.append(f"Accessible: {item['accessible']}")
        if item["pet_friendly"]:
            flags.append(f"Pet friendly: {item['pet_friendly']}")
        if flags:
            lines.append(f"   {', '.join(flags)}")
        if item["phone"]:
            lines.append(f"   Phone: {item['phone']}")
        lines.append(f"   Map: {maps_link(item['lat'], item['lon'])}")

    lines.append(
        "Weekly hours, holiday schedules, one-off closures, and access policies can change. "
        "The City advises calling ahead before visiting."
    )
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="cool_options_lookup",
            description=(
                "Find live NYC Cool Options. Use kind='cooling_center' for activated centers, "
                "kind='indoor' for indoor A/C options, or kind='all' for any heat-relief option. "
                "Pass `on` when the resident asks about a specific day or date; the result will "
                "rank and report that date's City-listed weekly schedule instead of today's status. "
                "Use audience='not_age_restricted' when the resident needs options without a City "
                "age restriction, including when asking for children."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string", "description": "NYC address or landmark"},
                    "site": {
                        "type": "string",
                        "description": (
                            "Optional exact facility name when checking hours for a site already "
                            "selected in the conversation; preserves that destination instead of reranking."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(_KINDS),
                        "default": "all",
                        "description": "Type of heat-relief location requested",
                    },
                    "audience": {
                        "type": "string",
                        "enum": list(_AUDIENCES),
                        "default": "any",
                        "description": (
                            "Use not_age_restricted when the resident needs options without a "
                            "City-listed age restriction. This does not prove child-specific "
                            "amenities or suitability."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Maximum results; use the user's requested count",
                    },
                    "on": {
                        "type": "string",
                        "format": "date",
                        "description": (
                            "Optional visit date in YYYY-MM-DD. Pass this when the resident names "
                            "a day or date; omit only for a current right-now lookup."
                        ),
                    },
                },
                "required": ["near"],
            },
            handler=_cool_options_lookup,
            open_world=True,
            title="Find Cool Options",
        )
    ]
