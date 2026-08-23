"""Useful public-restroom lookup backed by official NYC sources."""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from pydantic import Field

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.temporal import parse_clock_minutes, weekly_open_status
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, normalize, query_dataset, row_url
from heynyc.core.tools.geo import (
    GeoPoint,
    format_distance,
    haversine_m,
    maps_link,
    miles,
    rank_nearby,
    resolve_location,
)

COOL_OPTIONS_URL = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "Cool_Options/FeatureServer/0"
)
_NYC_TZ = ZoneInfo("America/New_York")
_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MATCH_RADIUS_M = 100
_LOCATION_SUFFIX_TOKENS = {
    "bronx", "brooklyn", "city", "island", "manhattan", "new", "ny", "queens", "staten",
    "york",
}


class PublicRestroomQuery(LocationRequest):
    near: str = Field(description="NYC address, neighborhood, or landmark to search near.")
    max_results: int | None = Field(
        default=None, ge=1, le=10, description="Maximum restrooms requested; omit for the default 3."
    )
    visit_date: date | None = Field(default=None, description="Requested New York visit date.")
    visit_time: time | None = Field(default=None, description="Requested New York visit time.")
    fully_accessible: bool = Field(
        default=False,
        description=(
            "Set true when the resident requests accessibility. This filters the City listing's "
            "site accessibility field; it does not prove restroom fixture accessibility."
        ),
    )
    changing_station: bool = Field(
        default=False,
        description="Set true only when the resident requests a changing station.",
    )


def _nyc_now() -> datetime:
    return datetime.now(_NYC_TZ)


def _open_now(record: dict, now: datetime) -> bool | None:
    schedule = {
        weekday: [
            (opened, closed)
            for interval in (1, 2)
            if (opened := parse_clock_minutes(record.get(f"cc_{day}_open{interval}")))
            is not None
            and (closed := parse_clock_minutes(record.get(f"cc_{day}_close{interval}")))
            is not None
        ]
        for weekday, day in enumerate(_DAYS)
    }
    return weekly_open_status(schedule, now)


def _scheduled_on(record: dict, requested: date) -> bool | None:
    day = _DAYS[requested.weekday()]
    known = False
    for weekday in _DAYS:
        for interval in (1, 2):
            opened = parse_clock_minutes(record.get(f"cc_{weekday}_open{interval}"))
            closed = parse_clock_minutes(record.get(f"cc_{weekday}_close{interval}"))
            if opened is None or closed is None:
                continue
            known = True
            if weekday == day:
                return True
    return False if known else None


def _name_tokens(value: object) -> set[str]:
    ignored = {
        "apt", "bathroom", "bpl", "facility", "nyc", "nypl", "pops", "public", "qpl",
        "restroom", "the",
    }
    return {token for token in re.findall(r"[a-z0-9]+", str(value).lower()) if token not in ignored}


def _is_named_origin(place_name: object, query: object) -> bool:
    place_tokens = _name_tokens(place_name)
    query_tokens = _name_tokens(query)
    return place_tokens == query_tokens or (
        len(place_tokens) >= 2
        and place_tokens < query_tokens
        and query_tokens - place_tokens <= _LOCATION_SUFFIX_TOKENS
    )


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
                "origin_label": origin.label,
                "point": [place.lat, place.lon],
                "distance_mi": distance_mi,
            },
        ),
    )


def _cool_citation(ctx: ToolContext, record: dict, origin: GeoPoint) -> str:
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
        provenance=data_provenance(
            record,
            record_id=record_id,
            field_pointer="/",
            derivation={"origin_label": origin.label},
        ),
    )


async def _public_restroom_lookup(args: dict, ctx: ToolContext) -> str:
    query = PublicRestroomQuery.model_validate(args)
    near = query.near.strip()
    now = _nyc_now()
    requested = query.visit_date
    if requested and requested < now.date():
        return (
            "The City sources provide current schedules, so I cannot verify a past service "
            "date. Ask for today or a future date."
        )
    future_schedule = requested is not None and requested > now.date()
    target_date = requested or now.date()
    target_at = (
        datetime.combine(target_date, query.visit_time, _NYC_TZ)
        if query.visit_time is not None
        else None
    )
    schedule_requested = future_schedule or target_at is not None

    binding = ctx.registry.dataset_bindings().get("public_restroom")
    if binding is None:
        return "The NYC public-restroom dataset is not configured."
    origin_result, city_result, cool_result = await asyncio.gather(
        resolve_location(near, ctx),
        query_dataset(binding.id, where=binding.where, limit=2000, client=ctx.http),
        query_feature_service(
            COOL_OPTIONS_URL,
            where="1=1" if schedule_requested else "Finder_status='OPEN'",
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
    exact_origin = next(
        (place for place in places if _is_named_origin(place.name, near)),
        None,
    )
    origin = (
        GeoPoint(
            exact_origin.lat,
            exact_origin.lon,
            exact_origin.name,
            confidence=1.0,
            match_type="dataset",
        )
        if exact_origin is not None
        else None if isinstance(origin_result, BaseException) else origin_result
    )
    if origin is None:
        return f"Could not locate '{near}'. Ask for a specific NYC address or landmark."
    if origin.low_confidence:
        return f"'{near}' may match several places. Ask for a specific NYC address or landmark."

    candidates = []
    partial_candidates = []
    fully_accessible = query.fully_accessible
    changing_station = query.changing_station
    for place, distance_m in rank_nearby(
        origin,
        places,
        key=lambda place: place.name.strip().casefold(),
    ):
        accessibility = str(place.raw.get("accessibility", "")).strip()
        station = str(place.raw.get("changing_stations", "")).strip().strip('"')
        access_match = accessibility.casefold() == "fully accessible"
        station_match = station.casefold().startswith("yes")
        missing = []
        unresolved = 0
        if fully_accessible and not access_match:
            unresolved += not accessibility
            missing.append(
                "site accessibility is not listed"
                if not accessibility
                else f"site accessibility is listed as {accessibility}"
            )
        if changing_station and not station_match:
            unresolved += not station
            missing.append(
                "changing-station availability is not listed"
                if not station
                else f"changing station is listed as {station}"
            )
        if missing:
            if (fully_accessible and access_match) or (changing_station and station_match):
                partial_candidates.append((-unresolved, len(missing), distance_m, place, missing))
            continue
        corroboration = _matching_cool_option(place, cool_records)
        open_now = (
            _open_now(corroboration, target_at)
            if corroboration and target_at is not None
            else
            _scheduled_on(corroboration, requested)
            if corroboration and future_schedule
            else _open_now(corroboration, now)
            if corroboration
            else None
        )
        evidence_rank = 0 if open_now is True else 2 if open_now is False else 1
        candidates.append((evidence_rank, distance_m, place, corroboration, open_now))

    if not candidates and not partial_candidates:
        return (
            "No NYC-listed public restroom matched the requested access features near "
            f"{origin.label}. Ask whether the resident wants the nearest result without those "
            "filters, or route them to 311."
        )

    limit = query.max_results or 3
    selected = (
        sorted(candidates, key=lambda item: (item[0], item[1]))
        if schedule_requested
        else candidates
    )[:limit]

    date_suffix = f" for {target_date.strftime('%A, %Y-%m-%d')}" if schedule_requested else ""
    if target_at is not None:
        date_suffix += f" at {target_at.strftime('%I:%M %p').lstrip('0')}"
    lines = [f"Public restrooms near {origin.label}{date_suffix}:"]
    if fully_accessible:
        lines.append(
            "The accessibility filter uses the City listing's site accessibility field; "
            "it does not prove restroom fixture accessibility."
        )
    if schedule_requested:
        lines.append(
            f"Ranked by requested-{'time' if target_at is not None else 'day'} schedule evidence, "
            "then distance, not by longest hours "
            "or restroom quality."
        )
    city_cites = []
    nearest_complete_m = min((item[1] for item in candidates), default=float("inf"))
    closer_partial = [
        item
        for item in sorted(partial_candidates, key=lambda item: item[:3])
        if item[2] < nearest_complete_m
    ][:limit]
    partial_evidence = {}
    if closer_partial and closer_partial[0][0] < 0 and ctx.toolbox and "web_fetch" in ctx.toolbox:
        _unresolved, _missing_count, _distance_m, place, missing = closer_partial[0]
        if place.website:
            query = (
                "site wheelchair accessibility"
                if any("accessibility" in item for item in missing)
                else "changing-station availability"
            )
            partial_evidence[place.record_id or place.name] = await ctx.toolbox[
                "web_fetch"
            ].handler({"url": place.website, "query": query}, ctx)
    classified_partial = []
    for item in closer_partial:
        _unresolved, _missing_count, _distance_m, place, missing = item
        evidence = partial_evidence.get(place.record_id or place.name, "")
        resolved = "site accessibility is not listed" in missing and any(
            line.strip().casefold() == "wheelchair accessible"
            for line in evidence.splitlines()
        )
        classified_partial.append((item, evidence, resolved))
    for resolved, heading in (
        (
            True,
            "Closest supported matches for the requested site features; restroom-fixture "
            "accessibility remains unverified:",
        ),
        (False, "Closer partial matches that do not satisfy every requested feature:"),
    ):
        group = [item for item in classified_partial if item[2] is resolved]
        if not group:
            continue
        lines.append(heading)
        for item, evidence, _resolved in group:
            _unresolved, _missing_count, distance_m, place, missing = item
            distance_mi = miles(distance_m)
            cite = _city_citation(ctx, binding, place, origin, distance_mi)
            city_cites.append(cite)
            lines.append(
                f"- {place.name}, {format_distance(near, origin, distance_mi, unit='miles', suffix='')} "
                f"{{cite:{cite}}}"
            )
            if not resolved:
                lines.append(f"  Missing constraint: {'; '.join(missing)} {{cite:{cite}}}")
            if place.website:
                lines.append(f"  Official facility page: {place.website} {{cite:{cite}}}")
            if evidence:
                lines.append(f"  Official page evidence: {evidence}")
                if resolved:
                    lines.append(
                        "  Official page resolves the missing site-accessibility field. Combined "
                        "with the City changing-station record, this is the closest supported "
                        "match for both requested site features. Restroom-fixture accessibility "
                        "remains unverified."
                    )
    if selected and fully_accessible:
        distance_label = "Farther " if closer_partial else ""
        station_label = " and a changing station" if changing_station else ""
        lines.append(
            f"{distance_label}City-dataset matches for site accessibility{station_label}; "
            "restroom-fixture accessibility is not verified:"
        )
    elif selected and closer_partial:
        lines.append("Farther City-listed matches:")
    for index, (_, distance_m, place, corroboration, open_now) in enumerate(selected, 1):
        distance_mi = miles(distance_m)
        city_cite = _city_citation(ctx, binding, place, origin, distance_mi)
        city_cites.append(city_cite)
        lines.append(
            f"{index}. {place.name}, "
            f"{format_distance(near, origin, distance_mi, unit='miles', suffix='')} {{cite:{city_cite}}}"
        )
        if corroboration:
            cool_cite = _cool_citation(ctx, corroboration, origin)
            day_name = _DAY_NAMES[target_date.weekday()]
            hours = str(corroboration.get(day_name, "")).strip()
            visit_label = (
                target_at.strftime("%I:%M %p").lstrip("0") if target_at is not None else ""
            )
            if target_at is not None and open_now is True:
                status = f"scheduled open at {visit_label} on {target_date.isoformat()}"
            elif target_at is not None and open_now is False:
                status = f"scheduled closed at {visit_label} on {target_date.isoformat()}"
            elif target_at is not None:
                status = f"schedule unclear at {visit_label} on {target_date.isoformat()}"
            elif future_schedule and open_now is True:
                status = (
                    f"site building is scheduled on {day_name}, {requested.isoformat()}"
                )
            elif future_schedule and open_now is False:
                status = f"no hours listed on {day_name}, {requested.isoformat()}"
            elif future_schedule:
                status = f"schedule unclear for {day_name}, {requested.isoformat()}"
            elif open_now is True:
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
            detail = (
                "   NYC lists this restroom, but its future building schedule is not "
                "independently corroborated"
                if schedule_requested
                else "   NYC lists this restroom, but it is not independently confirmed open now"
            )
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
            lines.append(
                f"   Official facility page: {place.website} {{cite:{city_cite}}}"
            )
        lines.append(
            f"   Map: {maps_link(place.lat, place.lon)} {{cite:{city_cite}}}"
        )

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
        "Current entry and restroom fixture condition were not verified by this lookup. "
        "If the first result does not work, try the next listed option."
    )
    if cool_failed:
        lines.append("The NYC Cool Options cross-check was unavailable for this lookup.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_public_restrooms",
            description=(
                "Find useful public restrooms near one NYC location. Cross-checks official "
                "Cool Options access and hours when the same site appears there."
            ),
            parameters=PublicRestroomQuery.model_json_schema(),
            handler=_public_restroom_lookup,
            open_world=True,
            title="Find public restrooms",
        )
    ]
