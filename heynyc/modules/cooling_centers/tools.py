"""Live lookup for NYC cooling centers and other Cool Options."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import geocode, haversine_m, maps_link, miles

COOL_OPTIONS_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Cool_Options/FeatureServer/0"
)
ACTIVE_CENTERS_URL = (
    "https://services5.arcgis.com/tMsas0Edz7Aih7fO/arcgis/rest/services/"
    "Cooling_Centers_PROD_view/FeatureServer/0"
)
_NYC_TZ = ZoneInfo("America/New_York")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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


def _item(record: dict, *, active: bool) -> dict | None:
    try:
        lat, lon = float(record["lat"]), float(record["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if active:
        item_type = "activated cooling center"
        source = ACTIVE_CENTERS_URL
    else:
        item_type = str(record.get("Space_type") or "cool option").strip().lower()
        source = COOL_OPTIONS_URL
    return {
        "record": record,
        "source": source,
        "name": _value(record, "Facility_name", "FACILITY_NAME"),
        "address": _value(record, "Address", "ADDRESS"),
        "phone": _value(record, "Phone", "PHONE"),
        "accessible": str(record.get("Accessible") or "").strip(),
        "pet_friendly": _value(record, "Pet_friendly", "PET_FRIENDLY"),
        "type": item_type,
        "lat": lat,
        "lon": lon,
        "active": active,
    }


async def _cool_options_lookup(args: dict, ctx: ToolContext) -> str:
    near = str(args.get("near", "")).strip()
    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
    if origin.low_confidence:
        return f"'{near}' may match several places. Ask for a specific NYC address or landmark."

    general_result, active_result = await asyncio.gather(
        query_feature_service(
            COOL_OPTIONS_URL,
            where="Finder_status='OPEN'",
            result_record_count=2000,
            client=ctx.http,
        ),
        query_feature_service(
            ACTIVE_CENTERS_URL,
            where="Finder_status='OPEN'",
            result_record_count=2000,
            client=ctx.http,
        ),
        return_exceptions=True,
    )
    general_failed = isinstance(general_result, BaseException)
    active_failed = isinstance(active_result, BaseException)
    general_records = [] if general_failed else general_result
    active_records = [] if active_failed else active_result

    kind = str(args.get("kind", "all"))
    items = [item for record in active_records if (item := _item(record, active=True))]
    if kind != "cooling_center":
        for record in general_records:
            if kind == "indoor" and str(record.get("Location_type", "")).lower() != "indoor":
                continue
            item = _item(record, active=False)
            if item:
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
            if active_failed:
                return (
                    "The NYC activated-center feed was unavailable, so I cannot safely say "
                    "whether a cooling center is activated. Try kind='indoor' for other Cool Options."
                )
            return (
                "No activated cooling centers were returned by the NYC finder. "
                "Try kind='indoor' for other indoor Cool Options."
            )
        if general_failed and active_failed:
            return "Both NYC Cool Options feeds were unavailable, so I cannot safely list locations."
        if general_failed:
            return (
                "The general Cool Options feed was unavailable, and the activated-center feed "
                "returned no locations. I cannot safely list other Cool Options."
            )
        return "No matching NYC Cool Options were found near that location."

    now = _nyc_now()
    for item in unique:
        item["open_now"] = _open_now(item["record"], now)
        item["distance_m"] = haversine_m(origin.lat, origin.lon, item["lat"], item["lon"])
    unique.sort(
        key=lambda item: (
            0 if item["open_now"] is True else 2 if item["open_now"] is False else 1,
            item["distance_m"],
        )
    )

    try:
        limit = int(args.get("limit", 3))
    except (TypeError, ValueError):
        limit = 3
    selected = unique[: min(max(limit, 1), 10)]
    day_name = _DAY_NAMES[now.weekday()]
    lines = [f"NYC Cool Options near {origin.label}:"]
    for index, item in enumerate(selected, 1):
        cite = _citation(ctx, item)
        distance = miles(item["distance_m"])
        lines.append(
            f"{index}. {item['name']}, {item['type']}, {distance:.2f} miles {{cite:{cite}}}"
        )
        if item["address"]:
            lines.append(f"   {item['address']}")
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
    lines.append("Hours and policies can change. The City advises calling ahead before visiting.")
    if general_failed:
        lines.append("The general Cool Options feed was unavailable for this lookup.")
    if active_failed:
        lines.append("The activated-center feed was unavailable for this lookup.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="cool_options_lookup",
            description=(
                "Find live NYC Cool Options. Use kind='cooling_center' for activated centers, "
                "kind='indoor' for indoor A/C options, or kind='all' for any heat-relief option."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string", "description": "NYC address or landmark"},
                    "kind": {
                        "type": "string",
                        "enum": ["all", "indoor", "cooling_center"],
                        "default": "all",
                        "description": "Type of heat-relief location requested",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Maximum results; use the user's requested count",
                    },
                },
                "required": ["near"],
            },
            handler=_cool_options_lookup,
            open_world=True,
            title="Find Cool Options",
        )
    ]
