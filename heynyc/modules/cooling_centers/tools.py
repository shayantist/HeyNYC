"""Live lookup for NYC cooling centers and other Cool Options."""
from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from heynyc.core.citations import content_hash, data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import ResidentFact, Tool, ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    current_resolved_location,
    directions_link,
    format_distance,
    geocode,
    haversine_m,
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
_INDOOR_NEXT_STEP = "Other indoor Cool Options can be checked."
_HEAT_HELP_URL = "https://portal.311.nyc.gov/article/?kanumber=KA-02663"
_HEAT_HELP = (
    "Call 311 for help finding a currently open place to cool down. "
    "If anyone has trouble breathing, call 911."
)
_SELECTED_INDOOR_UNAVAILABLE = (
    "The selected indoor Cool Option is not confirmed open now. The City result does not "
    "confirm usable air conditioning there now."
)


class CoolingQuery(BaseModel):
    """Validated resident constraints for one Cool Options lookup."""

    model_config = ConfigDict(extra="forbid")

    near: str = Field(description="NYC address or landmark.")
    site: str | None = Field(
        default=None,
        description="Exact facility name already selected in the conversation.",
    )
    exclude_sites: list[Annotated[str, Field(description="Exact rejected facility name.")]] = Field(
        default_factory=list,
        max_length=10,
        description="Exact facility names the resident rejected in this turn.",
    )
    kind: Literal["all", "indoor", "cooling_center"] = Field(
        default="all",
        description="Type of Cool Option requested by the resident.",
    )
    audience: Literal["any", "not_age_restricted"] = Field(
        default="any",
        description="Whether to exclude City rows marked as age-restricted.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Maximum locations requested by the resident. Set only when the resident explicitly "
            "asks for a number; otherwise omit it and the server returns three."
        ),
    )
    visit_date: date | None = Field(
        default=None,
        description="Visit date extracted from the resident's request.",
    )
    visit_time: time | None = Field(
        default=None,
        description="Visit time in America/New_York extracted from the resident's request.",
    )
    open_now_only: bool = Field(
        default=False,
        description="True only when the resident explicitly asks for a place open right now.",
    )


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


def _citation(ctx: ToolContext, item: dict, origin: GeoPoint) -> str:
    record = item["record"]
    record_id = str(record.get("OBJECTID", ""))
    url = feature_query_url(item["source"], record_id) if record_id else item["source"]
    return ctx.citations.register(
        url,
        snippet=f"{item['name']}, {item['type']}, status: {record.get('Finder_status', '')}",
        title="NYC Emergency Management Cool Options",
        kind="DATA",
        valid_as_of=_edit_date(record),
        provenance=data_provenance(
            record,
            record_id=record_id,
            field_pointer="/",
            derivation={
                "origin": [origin.lat, origin.lon],
                "origin_label": origin.label,
                **{
                    key: item[key]
                    for key in ("open_now", "target_at", "target_status")
                    if key in item
                },
            },
        ),
    )


def _terminal_citation(
    ctx: ToolContext,
    items: list[dict],
    now: datetime,
    snippet: str,
    derivation: dict | None = None,
) -> str:
    rows = [
        {
            "record_id": str(item["record"].get("NYCEM_ID") or item["record"].get("OBJECTID", "")),
            "snapshot": item["record"],
            "content_hash": content_hash(item["record"]),
            "open_now": item["open_now"],
        }
        for item in items
    ]
    valid_as_of = {_edit_date(item["record"]) for item in items}
    return ctx.citations.register(
        COOL_OPTIONS_URL,
        snippet=snippet,
        title="NYC Emergency Management Cool Options",
        kind="DATA",
        valid_as_of=valid_as_of.pop() if len(valid_as_of) == 1 else "",
        provenance=data_provenance(
            {"rows": rows},
            record_id="filtered-query",
            field_pointer="/rows",
            derivation={
                "now": now.isoformat(),
                "open_now": [
                    {"record_id": row["record_id"], "value": row["open_now"]}
                    for row in rows
                ],
                "next_open": [
                    {
                        "record_id": str(
                            item["record"].get("NYCEM_ID")
                            or item["record"].get("OBJECTID", "")
                        ),
                        "value": _next_open(item["record"], now),
                    }
                    for item in items
                ],
                **(derivation or {}),
            },
        ),
    )


def _heat_help_citation(ctx: ToolContext) -> str:
    return ctx.citations.register(
        _HEAT_HELP_URL,
        snippet=_HEAT_HELP,
        title="Cooling Centers, NYC311",
        kind="DOC",
        valid_as_of="2026-08-11",
        provenance={"snapshot": {"verified_fact": _HEAT_HELP}},
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


def _site_name(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _match_site(
    items: list[dict], requested: str = ""
) -> tuple[dict | None, set[str]]:
    requested_name = _site_name(requested)
    if not requested_name:
        return None, set()
    matches = [item for item in items if _site_name(item.get("name")) == requested_name]
    return (matches[0] if len(matches) == 1 else None), set()


def _site_from_turn(items: list[dict], turn: str, requested: str = "") -> dict | None:
    del turn
    return _match_site(items, requested)[0]


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


async def _find_cool_options(args: dict, ctx: ToolContext) -> str:
    query = CoolingQuery.model_validate(args)
    near = query.near.strip()
    current_turn = ctx.query
    site = (query.site or "").strip()
    offered_fact = ctx.resident_facts.get(_OFFERED_SITE_FACT)
    prior_offered = (
        _decode_site_fact(offered_fact.value, offered=True)
        if offered_fact
        else None
    )
    current_location = resident_supplied_location(
        near,
        current_turn,
        (),
    ) if current_turn else ""
    if site and prior_offered and not current_location:
        origin = ctx.current_location or GeoPoint(
            prior_offered[1][0], prior_offered[1][1], "previously resolved origin"
        )
    elif current_turn:
        stored_origin = current_resolved_location(near, ctx)
        near = (
            resident_supplied_location(
                near,
                current_turn,
                ctx.user_turns,
                allow_prior=True,
            )
            or (stored_origin.resident_query if stored_origin else "")
        )
        if not near:
            return (
                "A location is required before ranking nearby Cool Options. "
                "Ask for the resident's neighborhood, address, or landmark."
            )
        if stored_origin is None:
            ctx.current_location = None
        origin = stored_origin or await geocode(near, client=ctx.http)
    else:
        origin = await geocode(near, client=ctx.http)
    if origin is None:
        return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
    if origin.low_confidence:
        return f"'{near}' may match several places. Ask for a specific NYC address or landmark."
    if origin.resident_query:
        ctx.current_location = origin

    try:
        records = await query_feature_service(
            COOL_OPTIONS_URL,
            where="Finder_status='OPEN'",
            result_record_count=2000,
            client=ctx.http,
        )
    except Exception:
        return "The NYC Cool Options finder was unavailable, so I cannot safely list locations."

    kind = query.kind
    audience = query.audience
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
                + _INDOOR_NEXT_STEP
            )
        return "No matching NYC Cool Options were found near that location."

    origin_value = [origin.lat, origin.lon]
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
    excluded_names = {
        _site_name(name)
        for name in query.exclude_sites
        if _site_name(name)
    }
    excluded = {
        _site_key(item)
        for item in offered_items
        if _site_name(item.get("name")) in excluded_names
    }
    selected, _ = _match_site(
        [item for item in offered_items if _site_key(item) not in excluded],
        site,
    )
    selected_item = None
    if selected:
        selected_item = selected
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
                selected_item = stored_item
                unique = [stored_item]
            else:
                return "I couldn't re-confirm the cooling site you selected in the current City finder."
        elif selected_fact:
            ctx.resident_facts.pop(_SELECTED_SITE_FACT, None)
    if excluded:
        unique = [item for item in unique if _site_key(item) not in excluded]
    candidate_items = [item for item in offered_items if _site_key(item) not in excluded]

    now = _nyc_now()
    target_date = query.visit_date or now.date()
    target_time = query.visit_time
    target_at = (
        datetime.combine(target_date, target_time, _NYC_TZ)
        if target_time is not None
        else None
    )
    planning_ahead = query.visit_date is not None or query.visit_time is not None
    open_now_only = query.open_now_only
    target_day_name = _DAY_NAMES[target_date.weekday()]
    for item in candidate_items:
        item["open_now"] = _open_now(item["record"], now)
        item["target_hours"] = _scheduled_hours(item["record"], target_date.weekday())
        item["target_open"] = _scheduled_open(item["record"], target_date.weekday())
        if target_at is not None:
            item["target_at"] = target_at.isoformat()
            target_status = _open_now(item["record"], target_at)
            item["target_status"] = (
                False
                if target_status is None and item["target_open"] is False
                else target_status
            )
        item["distance_m"] = haversine_m(origin.lat, origin.lon, item["lat"], item["lon"])
    current_items = candidate_items
    current_open = [item for item in current_items if item["open_now"] is True]
    unavailable_note = ""
    if not planning_ahead and open_now_only:
        if selected_item and selected_item["open_now"] is not True:
            selected_label = {
                "cooling_center": "cooling center",
                "indoor": "indoor Cool Option",
            }.get(kind, "Cool Option")
            selected_cite = _citation(ctx, selected_item, origin)
            alternatives = [
                item
                for item in current_open
                if _site_key(item) != _site_key(selected_item)
            ]
            if alternatives:
                unique = alternatives
                unavailable_note = (
                    f"The selected {selected_label} is not confirmed open now. "
                    f"{{cite:{selected_cite}}} Current alternatives:"
                )
            elif kind == "cooling_center":
                return (
                    f"The selected {selected_label} is not confirmed open now. "
                    f"{{cite:{selected_cite}}} {_INDOOR_NEXT_STEP}"
                )
            else:
                heat_cite = _heat_help_citation(ctx)
                ctx.cooling_terminal_result = (
                    f"{_SELECTED_INDOOR_UNAVAILABLE} {_HEAT_HELP}"
                )
                ctx.cooling_terminal_citation_ids = (selected_cite, heat_cite)
                ctx.cooling_terminal_synthesis = True
                return (
                    f"Resolved origin: origin={origin.lat:.5f},{origin.lon:.5f}\n"
                    f"Selected-site status and A/C: {_SELECTED_INDOOR_UNAVAILABLE} "
                    f"{{cite:{selected_cite}}}\n"
                    f"NYC311 next step and emergency: {_HEAT_HELP} {{cite:{heat_cite}}}"
                )
        if selected_item is None and not current_open:
            if kind == "cooling_center":
                return f"No activated cooling center is confirmed open now. {_INDOOR_NEXT_STEP}"
            label = "indoor Cool Options" if kind == "indoor" else "Cool Options"
            absence = f"No current {label} are confirmed open now. I cannot safely recommend a destination."
            result = f"{absence} {_HEAT_HELP}"
            ctx.cooling_terminal_result = result
            ctx.cooling_terminal_citation_ids = (
                _terminal_citation(ctx, current_items, now, absence),
                _heat_help_citation(ctx),
            )
            return result
        if selected_item is None:
            unique = current_open
    if planning_ahead:
        unique.sort(
            key=lambda item: (
                0
                if item.get("target_status", item["target_open"]) is True
                else 2
                if item.get("target_status", item["target_open"]) is False
                else 1,
                item["distance_m"],
            )
        )
    elif open_now_only:
        unique.sort(
            key=lambda item: (
                0 if item["open_now"] is True else 2 if item["open_now"] is False else 1,
                item["distance_m"],
            )
        )
    else:
        unique.sort(key=lambda item: item["distance_m"])

    # F068: when the nearest open site is farther than sites that are closed right
    # now, say so in data terms so the answer is framed honestly instead of reading
    # "nearest option is a pet store 2 miles away" with no context.
    nearest_open = (
        None
        if planning_ahead
        else min(
            (item for item in current_items if item["open_now"] is True),
            key=lambda item: item["distance_m"],
            default=None,
        )
    )
    closer_closed = (
        [
            item
            for item in current_items
            if item["open_now"] is False and item["distance_m"] < nearest_open["distance_m"]
        ]
        if nearest_open
        else []
    )

    selected = unique[:query.max_results or 3]
    day_name = _DAY_NAMES[now.weekday()]
    target_date_label = (
        f"{target_date.strftime('%A, %B')} {target_date.day}, {target_date.year}"
    )
    date_scope = f" for {target_date_label}" if planning_ahead else ""
    lines = [
        f"NYC Cool Options{date_scope} near {origin.label}:",
    ]
    if planning_ahead:
        lines.append(
            "Activation status is current at lookup time, not a guarantee for the requested date."
        )
    if unavailable_note:
        lines.append(unavailable_note)
    if closer_closed:
        count = len(closer_closed)
        note = (
            f"{count} closer {'option is' if count == 1 else 'options are'} "
            "scheduled closed right now"
        )
        reopenings = [r for item in closer_closed if (r := _next_open(item["record"], now))]
        if reopenings:
            note += f"; the soonest reopens {min(reopenings)[2]}"
        aggregate_cite = _terminal_citation(
            ctx,
            closer_closed,
            now,
            note,
            derivation={
                "origin": [origin.lat, origin.lon],
                "predicate": (
                    "open_now is false and distance_m < nearest_open_distance_m"
                ),
                "nearest_open": {
                    "record_id": str(
                        nearest_open["record"].get("NYCEM_ID")
                        or nearest_open["record"].get("OBJECTID", "")
                    ),
                    "distance_m": nearest_open["distance_m"],
                },
                "distances_m": [
                    {
                        "record_id": str(
                            item["record"].get("NYCEM_ID")
                            or item["record"].get("OBJECTID", "")
                        ),
                        "value": item["distance_m"],
                    }
                    for item in closer_closed
                ],
            },
        )
        lines.append(f"{note}. {{cite:{aggregate_cite}}}")
    for index, item in enumerate(selected, 1):
        cite = _citation(ctx, item, origin)
        distance = format_distance(
            near,
            origin,
            miles(item["distance_m"]),
            unit="miles",
            suffix="",
        )
        item_type = (
            "cooling center"
            if planning_ahead and item["active"]
            else item["type"]
        )
        lines.append(
            f"{index}. {item['name']}, {item_type}, {distance} {{cite:{cite}}}"
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
        if target_at is not None:
            visit_time = target_at.strftime("%I:%M %p").lstrip("0")
            target_status = item["target_status"]
            if target_status is True:
                status = "scheduled open"
            elif target_status is False:
                status = "scheduled closed"
            else:
                status = "schedule unknown"
            lines.append(f"   {status} at {visit_time} America/New_York")
            if item["target_hours"]:
                lines.append(
                    f"   Scheduled {target_date_label}: {item['target_hours']}"
                )
        elif planning_ahead and item["target_hours"]:
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
        if item["accessible"]:
            lines.append(
                "   Step-free entrance: not confirmed by the City accessibility field"
            )
        if item["phone"]:
            lines.append(f"   Phone: {item['phone']}")
        destination = GeoPoint(item["lat"], item["lon"], item["name"])
        lines.append(f"   Directions from resolved origin: {directions_link(origin, destination)}")

    lines.append(
        "Weekly hours, holiday schedules, one-off closures, and access policies can change. "
        "Call ahead before visiting."
    )
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_cool_options",
            description=(
                "Find live NYC Cool Options. Use kind='cooling_center' for activated centers, "
                "kind='indoor' for indoor A/C options, or kind='all' for any heat-relief option. "
                "Pass `visit_date` when the resident asks about a specific day or date; the result will "
                "rank and report that date's City-listed weekly schedule instead of today's status. "
                "Pass `visit_time` when the resident names a visit time; the tool computes scheduled "
                "status in America/New_York. "
                "Use audience='not_age_restricted' when the resident needs options without a City "
                "age restriction, including when asking for children."
            ),
            parameters=CoolingQuery.model_json_schema(),
            handler=_find_cool_options,
            open_world=True,
            title="Find Cool Options",
        )
    ]
