"""Useful public-restroom lookup backed by official NYC sources."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, normalize, query_dataset, row_url
from heynyc.core.tools.geo import geocode, haversine_m, maps_link, miles

COOL_OPTIONS_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Cool_Options/FeatureServer/0"
)
_NYC_TZ = ZoneInfo("America/New_York")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MATCH_RADIUS_M = 100


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


def _name_tokens(value: object) -> set[str]:
    ignored = {
        "bathroom", "bpl", "facility", "nyc", "nypl", "pops", "public", "qpl",
        "restroom", "the",
    }
    return {token for token in re.findall(r"[a-z0-9]+", str(value).lower()) if token not in ignored}


def _matching_cool_option(place, records: list[dict]) -> dict | None:
    matches = []
    place_tokens = _name_tokens(place.name)
    for record in records:
        try:
            distance = haversine_m(place.lat, place.lon, float(record["lat"]), float(record["lon"]))
        except (KeyError, TypeError, ValueError):
            continue
        record_tokens = _name_tokens(record.get("Facility_name"))
        same_name = len(place_tokens) >= 2 and place_tokens == record_tokens
        if distance <= _MATCH_RADIUS_M and same_name:
            matches.append((distance, record))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _city_citation(ctx: ToolContext, binding, place, origin, distance_mi: float) -> str:
    url = row_url(binding.id, place.record_id) if place.record_id else place.source_url
    details = [
        str(place.raw.get(field, "")).strip()
        for field in (
            "open",
            "hours_of_operation",
            "accessibility",
            "restroom_type",
            "changing_stations",
            "additional_notes",
        )
        if str(place.raw.get(field, "")).strip()
    ]
    return ctx.citations.register(
        url,
        snippet=", ".join([f"{place.name}, status: {place.status}", *details]),
        title=f"NYC Open Data ({binding.id})",
        kind="DATA",
        valid_as_of=place.updated_at,
        provenance=data_provenance(
            place.raw,
            record_id=place.record_id,
            field_pointer="/",
            derivation={
                "origin": [origin.lat, origin.lon],
                "point": [place.lat, place.lon],
                "distance_mi": distance_mi,
            },
        ),
    )


def _cool_citation(ctx: ToolContext, record: dict) -> str:
    record_id = str(record.get("OBJECTID", ""))
    url = feature_query_url(COOL_OPTIONS_URL, record_id) if record_id else COOL_OPTIONS_URL
    return ctx.citations.register(
        url,
        snippet=(
            f"{record.get('Facility_name', '')}, {record.get('Space_type', '')}, "
            f"status: {record.get('Finder_status', '')}"
        ),
        title="NYC Emergency Management Cool Options",
        kind="DATA",
        provenance=data_provenance(record, record_id=record_id, field_pointer="/"),
    )


async def _public_restroom_lookup(args: dict, ctx: ToolContext) -> str:
    near = str(args.get("near", "")).strip()
    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
    if origin.low_confidence:
        return f"'{near}' may match several places. Ask for a specific NYC address or landmark."

    binding = ctx.registry.dataset_bindings().get("public_restroom")
    if binding is None:
        return "The NYC public-restroom dataset is not configured."

    city_result, cool_result = await asyncio.gather(
        query_dataset(binding.id, where=binding.where, limit=2000, client=ctx.http),
        query_feature_service(
            COOL_OPTIONS_URL,
            where="Finder_status='OPEN'",
            result_record_count=2000,
            client=ctx.http,
        ),
        return_exceptions=True,
    )
    if isinstance(city_result, BaseException):
        return "The NYC public-restroom feed is unavailable, so I cannot safely list locations."
    city_records = city_result
    cool_failed = isinstance(cool_result, BaseException)
    cool_records = [] if cool_failed else cool_result
    places = normalize(city_records, binding.field_map, source_url=dataset_url(binding.id))
    if not places:
        return "No public restrooms were found in the NYC dataset."

    now = _nyc_now()
    candidates = []
    seen: set[str] = set()
    fully_accessible = args.get("fully_accessible") is True
    changing_station = args.get("changing_station") is True
    for place in sorted(
        places,
        key=lambda item: haversine_m(origin.lat, origin.lon, item.lat, item.lon),
    ):
        if fully_accessible and str(
            place.raw.get("accessibility", "")
        ).strip().casefold() != "fully accessible":
            continue
        if changing_station and not str(
            place.raw.get("changing_stations", "")
        ).strip().strip('"').casefold().startswith("yes"):
            continue
        key = place.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        distance_m = haversine_m(origin.lat, origin.lon, place.lat, place.lon)
        corroboration = _matching_cool_option(place, cool_records)
        open_now = _open_now(corroboration, now) if corroboration else None
        evidence_rank = 0 if open_now is True else 2 if open_now is False else 1
        candidates.append((evidence_rank, distance_m, place, corroboration, open_now))

    if not candidates:
        return (
            "No NYC-listed public restroom matched the requested access features near "
            f"{origin.label}. Ask whether the resident wants the nearest result without those "
            "filters, or route them to 311."
        )

    try:
        limit = int(args.get("limit", 3))
    except (TypeError, ValueError):
        limit = 3
    limit = min(max(limit, 1), 10)
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:limit]

    lines = [f"Public restrooms near {origin.label}:"]
    city_cites = []
    for index, (_, distance_m, place, corroboration, open_now) in enumerate(selected, 1):
        distance_mi = miles(distance_m)
        city_cite = _city_citation(ctx, binding, place, origin, distance_mi)
        city_cites.append(city_cite)
        lines.append(f"{index}. {place.name}, {distance_mi:.2f} miles {{cite:{city_cite}}}")
        if corroboration:
            cool_cite = _cool_citation(ctx, corroboration)
            day_name = _DAY_NAMES[now.weekday()]
            hours = str(corroboration.get(day_name, "")).strip()
            if open_now is True:
                status = "scheduled open now"
            elif open_now is False:
                status = "scheduled closed now"
            else:
                status = "listed in NYC Cool Options, current hours unclear"
            detail = f"   {status}"
            if hours:
                detail += f". {day_name}: {hours}"
            detail += (
                ". This corroborates access to the site but does not confirm the restroom's "
                f"condition {{cite:{cool_cite}}}"
            )
            lines.append(detail)
            accessible = str(corroboration.get("Accessible", "")).strip()
            if accessible:
                lines.append(
                    f"   Wheelchair accessible: {accessible} for the site; restroom fixtures "
                    f"are not confirmed {{cite:{cool_cite}}}"
                )
        else:
            hours = str(place.raw.get("hours_of_operation", "")).strip()
            detail = "   NYC lists this restroom, but it is not independently confirmed open now"
            if hours:
                detail += f". Listed hours: {hours}"
            lines.append(f"{detail} {{cite:{city_cite}}}")
        season = str(place.raw.get("open", "")).strip()
        accessibility = str(place.raw.get("accessibility", "")).strip()
        restroom_type = str(place.raw.get("restroom_type", "")).strip()
        changing_station = str(place.raw.get("changing_stations", "")).strip()
        access_note = str(place.raw.get("additional_notes", "")).strip()
        if season:
            lines.append(f"   Seasonal availability: {season} {{cite:{city_cite}}}")
        if accessibility:
            lines.append(
                f"   NYC listing accessibility: {accessibility} {{cite:{city_cite}}}"
            )
        if restroom_type:
            lines.append(f"   Restroom type: {restroom_type} {{cite:{city_cite}}}")
        if changing_station:
            lines.append(f"   Changing station: {changing_station} {{cite:{city_cite}}}")
        if access_note:
            lines.append(f"   Access note: {access_note} {{cite:{city_cite}}}")
        if place.website:
            lines.append(f"   Official facility page: {place.website}")
        lines.append(f"   Map: {maps_link(place.lat, place.lon)}")

    record_dates = sorted({item[2].updated_at[:10] for item in selected if item[2].updated_at})
    record_cites = " ".join(f"{{cite:{citation_id}}}" for citation_id in city_cites)
    if len(record_dates) == 1:
        lines.append(f"NYC restroom record date: {record_dates[0]}. {record_cites}")
    elif record_dates:
        lines.append(
            f"NYC restroom record dates: {record_dates[0]} through {record_dates[-1]}. "
            f"{record_cites}"
        )
    lines.append(
        "NYC restroom records are not real-time. For a locked, closed, or unusable restroom, "
        "try the next result or report the problem to 311."
    )
    if cool_failed:
        lines.append("The NYC Cool Options cross-check was unavailable for this lookup.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="public_restroom_lookup",
            description=(
                "Find useful public restrooms near one NYC location. Cross-checks official "
                "Cool Options access and hours when the same site appears there."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string", "description": "NYC address or landmark"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Maximum results to return; use the user's requested count",
                    },
                    "fully_accessible": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Set true only when the resident requests a fully accessible restroom"
                        ),
                    },
                    "changing_station": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Set true only when the resident requests a changing station"
                        ),
                    },
                },
                "required": ["near"],
            },
            handler=_public_restroom_lookup,
            open_world=True,
            title="Find public restrooms",
        )
    ]
