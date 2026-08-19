"""Read-only 311 service-request tools grounded in NYC Open Data erm2-nwe9.

  - "is my complaint moving?"  -> sr_number: look up one row by its own unique_key
    (the SR number). ONLY the number is sent; a resident's address is never geocoded
    or logged on this path.
  - "what's happening with 311 complaints about X near me?" -> complaint_terms / near: recent
    rows filtered on the dataset's own columns (complaint_type, created_date) and the
    location point column (within_circle), summarized by status.

Every fact is cited with a full-row DATA provenance snapshot and a re-fetchable
permalink keyed on the row's own unique_key; valid_as_of comes from the record's own
dates. Filing (Create-SR) is out of scope: this lane cannot change a 311 request.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset
from heynyc.core.tools.geo import GeoPoint, geocode, maps_link

# ponytail: single source for the dataset id; the manifest declares the matching binding
# (category service_request_311) only so the capability table / grounding label read it.
DATASET_ID = "erm2-nwe9"
_NYC_TZ = ZoneInfo("America/New_York")
_DEFAULT_WITHIN_DAYS = 30
_MIN_WITHIN_DAYS = 1
_MAX_WITHIN_DAYS = 365
_DEFAULT_RADIUS_METERS = 800
_MIN_RADIUS_METERS = 100
_MAX_RADIUS_METERS = 50_000
_AREA_ORDER = "created_date DESC"
_AREA_SELECT = "status, count(*) as count"
_REFILE_URL = "https://portal.311.nyc.gov/article/?kanumber=KA-02419"
_REFILE_VERIFIED_ON = "2026-07-26"
_REFILE_GUIDANCE = (
    "If a closed service request is still unresolved, refile the service request. "
    "NYC311 feedback does not address an individual service request."
)


def _nyc_now() -> datetime:
    return datetime.now(_NYC_TZ)


def _sr_permalink(unique_key: str) -> str:
    """Re-fetchable single-row permalink keyed on the row's OWN unique_key.

    erm2-nwe9's `:id` row-permalink endpoint 404s (newer row-id format), so the stable,
    re-fetchable locator is a filtered query on the SR's own number, verified live."""
    return dataset_url(DATASET_ID) + "?$where=" + quote(f"unique_key='{unique_key}'", safe="")


def _valid_as_of(record: dict) -> str:
    """Temporal provenance from the record's OWN dates: when this SR last moved."""
    for key in ("resolution_action_updated_date", "closed_date", "created_date"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _complaint_label(record: dict) -> str:
    complaint = str(record.get("complaint_type", "") or "").strip()
    descriptor = str(record.get("descriptor", "") or "").strip()
    return f"{complaint} ({descriptor})" if complaint and descriptor else complaint or descriptor


def _sr_citation(ctx: ToolContext, record: dict) -> str:
    key = str(record.get("unique_key", "") or "").strip()
    status = str(record.get("status", "") or "").strip()
    opened = str(record.get("created_date", "") or "")[:10]
    resolution = str(record.get("resolution_description", "") or "").strip()
    bits = [f"SR {key}", _complaint_label(record), f"status {status}", f"opened {opened}"]
    if resolution:
        bits.append(resolution)
    return ctx.citations.register(
        _sr_permalink(key),
        snippet="; ".join(bit for bit in bits if bit.strip()),
        title=f"NYC 311 Service Request {key}",
        kind="DATA",
        valid_as_of=_valid_as_of(record),
        # Full-row snapshot: the whole record the model may describe, hashed and re-fetchable.
        provenance=data_provenance(record, record_id=key, field_pointer="/"),
    )


def _refile_citation(ctx: ToolContext) -> str:
    return ctx.citations.register(
        _REFILE_URL,
        snippet=_REFILE_GUIDANCE,
        title="NYC311 Feedback",
        kind="DOC",
        valid_as_of=_REFILE_VERIFIED_ON,
    )


def _empty_area_citation(
    ctx: ToolContext,
    where: str,
    scope: str,
    origin: GeoPoint | None,
) -> str:
    checked_at = _nyc_now().isoformat()
    url = _area_query_url(where)
    return ctx.citations.register(
        url,
        snippet=f"No matching 311 rows for {scope}; checked {checked_at}",
        title="NYC 311 Service Requests search",
        kind="DATA",
        valid_as_of=checked_at,
        provenance=data_provenance(
            {"status_counts": {}, "examples": []},
            record_id="empty-area-search",
            field_pointer="/",
            derivation=_area_derivation(where, origin=origin, checked_at=checked_at),
        ),
    )


def _area_citation(
    ctx: ToolContext,
    where: str,
    scope: str,
    rows: list[dict],
    counts: Counter,
    max_results: int,
    origin: GeoPoint | None,
) -> str:
    checked_at = _nyc_now().isoformat()
    breakdown = ", ".join(f"{n} {status}" for status, n in counts.most_common())
    total = sum(counts.values())
    return ctx.citations.register(
        _area_query_url(where),
        snippet=(
            f"{total} recent 311 rows for {scope}; {total} total matches; "
            f"{len(rows)} most recent examples shown; status: {breakdown}"
        ),
        title="NYC 311 Service Requests search",
        kind="DATA",
        valid_as_of=checked_at,
        provenance=data_provenance(
            {"status_counts": dict(counts), "examples": rows},
            record_id="area-search",
            field_pointer="/",
            derivation=_area_derivation(
                where,
                max_results,
                origin=origin,
                checked_at=checked_at,
            ),
        ),
    )


def _area_derivation(
    where: str,
    max_results: int | None = None,
    *,
    origin: GeoPoint | None = None,
    checked_at: str | None = None,
) -> dict:
    derivation = {
        "where": where,
        "aggregate": {"select": _AREA_SELECT, "group": "status"},
    }
    if origin is not None:
        derivation["origin"] = {
            "label": origin.label,
            "latitude": origin.lat,
            "longitude": origin.lon,
        }
    if checked_at is not None:
        derivation["checked_at"] = checked_at
    if max_results is not None:
        derivation["examples"] = {
            "order": _AREA_ORDER,
            "limit": max_results,
            "exclude_system_fields": False,
        }
    return derivation


def _area_query_url(where: str) -> str:
    return dataset_url(DATASET_ID) + "?" + urlencode({
        "$where": where,
        "$select": _AREA_SELECT,
        "$group": "status",
    })


def _approximate_miles(radius_meters: int) -> str:
    miles = round(radius_meters / 1609.344, 1)
    amount = str(int(miles)) if miles.is_integer() else str(miles)
    return f"about {amount} {'mile' if miles <= 1 else 'miles'}"


async def _lookup_sr(sr_number: str, ctx: ToolContext) -> str:
    # PII/injection boundary: reduce to digits, so ONLY the number reaches Socrata and a stray
    # quote can never alter the SoQL. NYC SR numbers are all digits (verified live).
    key = re.sub(r"\D", "", sr_number)
    if not key:
        return (
            "That does not look like a 311 service request number. It is all digits, like 69741503. "
            "You can find it on your 311 confirmation, or check status at portal.311.nyc.gov."
        )
    rows = await query_dataset(DATASET_ID, where=f"unique_key='{key}'", limit=1, client=ctx.http)
    if not rows:
        return (
            f"I could not find a NYC 311 service request with number {key}. Double-check the number "
            "on your 311 confirmation. You can also see the official status at portal.311.nyc.gov or "
            "by calling 311."
        )
    record = rows[0]
    status = str(record.get("status", "") or "").strip() or "Unknown"
    opened = str(record.get("created_date", "") or "")[:10]
    closed = str(record.get("closed_date", "") or "")[:10]
    agency = str(record.get("agency_name", "") or record.get("agency", "") or "").strip()
    resolution = str(record.get("resolution_description", "") or "").strip()
    cite = _sr_citation(ctx, record)

    lines = [
        f"NYC 311 Service Request {key}: {_complaint_label(record) or 'complaint'}",
        f"- Status: {status} {{cite:{cite}}}",
        f"- Opened {opened}" + (f", closed {closed}" if closed else ""),
    ]
    if agency:
        lines.append(f"- Handled by: {agency}")
    if resolution:
        lines.append(f"- Resolution on record: {resolution}")
    if status.lower() == "closed":
        refile_cite = _refile_citation(ctx)
        lines.append(
            f"This request is marked closed. {_REFILE_GUIDANCE} {{cite:{refile_cite}}}"
        )
    else:
        as_of = _valid_as_of(record)[:10]
        lines.append(
            f"This request is still open (status {status})"
            + (f"; its last recorded update was {as_of}." if as_of else ".")
        )
    return "\n".join(lines)


async def _lookup_area(
    complaint_terms: list[str],
    near: str,
    max_results: int,
    within_days: int,
    radius_meters: int,
    ctx: ToolContext,
) -> str:
    cutoff = (
        (_nyc_now().astimezone(UTC) - timedelta(days=within_days))
        .astimezone(_NYC_TZ)
        .strftime("%Y-%m-%dT%H:%M:%S")
    )
    clauses = [f"created_date > '{cutoff}'"]
    if complaint_terms:
        matches = []
        for term in complaint_terms:
            safe = term.replace("'", "''")  # SoQL escapes a quote by doubling it
            matches.extend([
                f"upper(complaint_type) like upper('%{safe}%')",
                f"upper(descriptor) like upper('%{safe}%')",
            ])
        clauses.append(f"({' OR '.join(matches)})")

    origin = None
    if near:
        # PII: geocode the location to bound the query. The resolved address is NEVER logged.
        origin = await geocode(near, client=ctx.http)
        if origin is None:
            return f"I could not locate '{near}'. Give me a specific NYC address or landmark."
        if origin.low_confidence:
            return (
                f"'{near}' could match several places. Give me a specific NYC address or landmark "
                "and I'll check 311 activity there."
            )
        clauses.append(
            f"within_circle(location, {origin.lat}, {origin.lon}, {radius_meters})"
        )

    where = " AND ".join(clauses)
    aggregate_rows = await query_dataset(
        DATASET_ID,
        where=where,
        select=_AREA_SELECT,
        group="status",
        limit=None,
        exclude_system_fields=None,
        client=ctx.http,
    )
    counts = Counter()
    for row in aggregate_rows:
        status = str(row.get("status", "") or "").strip() or "Unknown"
        try:
            count = int(row.get("count", 0))
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[status] += count

    scope = (
        "about " + " or ".join(f'\"{term}\"' for term in complaint_terms) + " "
        if complaint_terms
        else ""
    )
    where_label = (
        f"within {radius_meters} meters ({_approximate_miles(radius_meters)}) of {origin.label}"
        if origin
        else "in that area"
    )
    if not counts:
        cite = _empty_area_citation(
            ctx,
            where,
            f'{scope}{where_label} in the last {within_days} days',
            origin,
        )
        return (
            f"I found no recent NYC 311 complaints {scope}{where_label} in the last {within_days} "
            f"days. That could mean none were filed, or they are older than this window. {{cite:{cite}}} "
            "To report a "
            "new problem or check official status, use portal.311.nyc.gov or call 311."
        )

    rows = await query_dataset(
        DATASET_ID,
        where=where,
        order=_AREA_ORDER,
        limit=max_results,
        client=ctx.http,
    )
    breakdown = ", ".join(f"{n} {status}" for status, n in counts.most_common())
    area_cite = _area_citation(
        ctx,
        where,
        f"{scope}{where_label} in the last {within_days} days".strip(),
        rows,
        counts,
        max_results,
        origin,
    )
    total = sum(counts.values())
    lines = [
        f"Recent NYC 311 complaints {scope}{where_label} "
        f"(created in the last {within_days} days) {{cite:{area_cite}}}:",
        f"- Of the {total} I found, status is: {breakdown} {{cite:{area_cite}}}",
        "Most recent:",
    ]
    for index, record in enumerate(rows[:max_results], 1):
        key = str(record.get("unique_key", "") or "").strip()
        status = str(record.get("status", "") or "").strip() or "Unknown"
        opened = str(record.get("created_date", "") or "")[:10]
        cite = _sr_citation(ctx, record)
        lines.append(
            f"{index}. SR {key}: {_complaint_label(record) or 'complaint'}, status {status}, "
            f"opened {opened} {{cite:{cite}}}"
        )
        try:
            lat, lon = float(record["latitude"]), float(record["longitude"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            lines.append(f"   Map: {maps_link(lat, lon)}")
    return "\n".join(lines)


async def _check_311_request(args: dict, ctx: ToolContext) -> str:
    return await _lookup_sr(str(args.get("sr_number", "") or "").strip(), ctx)


def _bounded_int(args: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


async def _search_311_complaints(args: dict, ctx: ToolContext) -> str:
    raw_terms = [
        str(term).strip()
        for term in (args.get("complaint_terms") or [])[:3]
        if str(term).strip()
    ]
    complaint_terms = [
        term for term in raw_terms if 4 <= len(term) <= 40
    ]
    if raw_terms and len(complaint_terms) != len(raw_terms):
        return "Use one to three complaint-type or descriptor terms of at least 4 characters each."
    near = str(args.get("near", "") or "").strip()
    if not complaint_terms and not near:
        return (
            "Tell me your 311 service request number to check its status, or a complaint type and an "
            "NYC location (like 'noise complaints near Union Square') to see recent 311 activity."
        )
    max_results = _bounded_int(args, "max_results", 5, 1, 10)
    within_days = _bounded_int(
        args,
        "within_days",
        _DEFAULT_WITHIN_DAYS,
        _MIN_WITHIN_DAYS,
        _MAX_WITHIN_DAYS,
    )
    radius_meters = _bounded_int(
        args,
        "radius_meters",
        _DEFAULT_RADIUS_METERS,
        _MIN_RADIUS_METERS,
        _MAX_RADIUS_METERS,
    )
    return await _lookup_area(
        complaint_terms,
        near,
        max_results,
        within_days,
        radius_meters,
        ctx,
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="check_311_request",
            description=(
                "Check one existing NYC 311 service request by its number. Send only the number. "
                "Read-only: reports status and cannot file, update, or close a request."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sr_number": {
                        "type": "string",
                        "description": "The resident's 311 service request number (digits only).",
                    },
                },
                "required": ["sr_number"],
            },
            handler=_check_311_request,
            open_world=True,  # hits live Socrata
            title="Check 311 request status",
        ),
        Tool(
            name="search_311_complaints",
            description=(
                "Search recent NYC 311 complaints by problem and optional NYC location. Use for "
                "area activity, not for checking one known service request. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "complaint_terms": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "minLength": 4,
                            "maxLength": 40,
                            "description": "One short complaint-type or descriptor term",
                        },
                        "description": (
                            "One to three short, dataset-facing alternative terms matched "
                            "independently against complaint type and descriptor. Prefer a stable "
                            "category word, such as ['rodent'] for rats. Do not pass a sentence, "
                            "combine alternatives into one string, or use a 2-3 letter fragment."
                        ),
                    },
                    "near": {
                        "type": "string",
                        "description": "Optional NYC address or landmark to bound the search",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Maximum recent complaints to list",
                    },
                    "within_days": {
                        "type": "integer",
                        "minimum": _MIN_WITHIN_DAYS,
                        "maximum": _MAX_WITHIN_DAYS,
                        "default": _DEFAULT_WITHIN_DAYS,
                        "description": "Lookback window in elapsed days; omit to use 30 days",
                    },
                    "radius_meters": {
                        "type": "integer",
                        "minimum": _MIN_RADIUS_METERS,
                        "maximum": _MAX_RADIUS_METERS,
                        "default": _DEFAULT_RADIUS_METERS,
                        "description": "Search radius around near, in meters; omit to use 800 meters",
                    },
                },
                "required": ["complaint_terms"],
            },
            handler=_search_311_complaints,
            open_world=True,
            title="Search recent 311 complaints",
        ),
    ]
