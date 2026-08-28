"""DOT street-closure lane, grounded in NYC Open Data i6b5-j7bu.

One resident intent: "are streets closed near <place> on <date>?" for game-day
or travel planning. This is DOT's schedule of construction street closures (by
block), so each row carries the closed segment (on/from/to street), the work
window (work_start_date -> work_end_date), the purpose, and a line geometry.

The handler geocodes `near` to a point, then filters on the dataset's OWN
columns, never on prose:
  - within_circle(the_geom, lat, lon, radius): the closure line runs near the
    resident's point (verified live: within_circle works on this MultiLineString),
  - work_start_date <= <day> AND work_end_date >= <day>: the closure is active on
    the asked date (default today, NYC time).

Every listed closure is cited with a full-row DATA provenance snapshot and a
re-fetchable permalink keyed on the row's own uniqueid; valid_as_of comes from
the record's own update date. This lane is DOT's PLANNED construction schedule:
it will not show a sudden incident, a parade, or a police closure, so it routes
live-traffic questions to 511NY / nyc.gov/dot. Alternate-side-parking suspension
status and the DOT traffic-advisory feed are not keyless (they sit behind the
api.nyc.gov APIM key), so they are out of scope for this lane, not faked.
"""
from __future__ import annotations

from datetime import date, datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

from pydantic import Field, ValidationError

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset, query_dataset_pages
from heynyc.core.tools.geo import maps_link, resolve_location

# ponytail: single source for the dataset id; the manifest declares the matching
# binding (category street_closure) only so the capability table / grounding label read it.
DATASET_ID = "i6b5-j7bu"
_NYC_TZ = ZoneInfo("America/New_York")
_RADIUS_M = 800   # ~0.5 mile, the bound handed to Socrata within_circle
_PAGE_SIZE = 200

_ROUTE = (
    "This is DOT's planned construction-closure schedule, so it will not show a sudden crash, a "
    "parade, or a police closure. For live traffic and incidents check 511NY (511ny.org) or "
    "nyc.gov/dot, and dial 511."
)


class StreetClosureQuery(LocationRequest):
    near: str = Field(description="NYC address or landmark to check closures around.")
    visit_date: date | None = Field(
        default=None, description="Date to check; omit to use today's New York date."
    )
    max_results: int | None = Field(
        default=None, ge=1, description="Maximum closures requested; omit for the default 5."
    )


def _nyc_today() -> date:
    return datetime.now(_NYC_TZ).date()


def _closure_permalink(uniqueid: str) -> str:
    """Re-fetchable single-row permalink keyed on the row's OWN uniqueid field (verified live)."""
    return dataset_url(DATASET_ID) + "?$where=" + quote(f"uniqueid='{uniqueid}'", safe="")


def _valid_as_of(record: dict) -> str:
    """Temporal provenance from the record's own dates: when DOT last refreshed this row,
    falling back to the closure's own scheduled start."""
    for key in (":updated_at", "work_start_date"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _representative_point(record: dict) -> tuple[float, float] | None:
    """A single (lat, lon) for the map link, read from the closure's own line geometry.

    the_geom is a GeoJSON MultiLineString ([[[lon, lat], ...], ...]); we take the first
    vertex. Returns None (never a fabricated coordinate) when the geometry is missing or
    malformed."""
    geom = record.get("the_geom")
    if not isinstance(geom, dict):
        return None
    try:
        lon, lat = geom["coordinates"][0][0][:2]
        return float(lat), float(lon)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _closure_citation(ctx: ToolContext, record: dict) -> str:
    uniqueid = str(record.get("uniqueid", "") or "").strip()
    on = str(record.get("onstreetname", "") or "").strip()
    frm = str(record.get("fromstreetname", "") or "").strip()
    to = str(record.get("tostreetname", "") or "").strip()
    purpose = str(record.get("purpose", "") or "").strip()
    start = str(record.get("work_start_date", "") or "")[:10]
    end = str(record.get("work_end_date", "") or "")[:10]
    segment = f"{on} from {frm} to {to}".strip()
    return ctx.citations.register(
        _closure_permalink(uniqueid),
        snippet=f"{segment}; {purpose}; closed {start} to {end}".strip("; "),
        title=f"NYC Open Data street closure ({DATASET_ID})",
        kind="DATA",
        valid_as_of=_valid_as_of(record),
        # Full-row snapshot: the whole record the model may describe, hashed and re-fetchable.
        provenance=data_provenance(record, record_id=uniqueid, field_pointer="/"),
    )


def _closure_line(index: int, record: dict, cite: str) -> str:
    on = str(record.get("onstreetname", "") or "").strip() or "a street"
    frm = str(record.get("fromstreetname", "") or "").strip()
    to = str(record.get("tostreetname", "") or "").strip()
    purpose = str(record.get("purpose", "") or "").strip() or "construction"
    start = str(record.get("work_start_date", "") or "")[:10]
    end = str(record.get("work_end_date", "") or "")[:10]
    where = f"{on}" + (f" between {frm} and {to}" if frm and to else "")
    parts = [
        f"{index}. {where}: {purpose}, scheduled {start} to {end} {{cite:{cite}}}"
    ]
    point = _representative_point(record)
    if point is not None:
        parts.append(f"   Map: {maps_link(point[0], point[1])}")
    return "\n".join(parts)


async def _street_closures(args: StreetClosureQuery, ctx: ToolContext) -> str:
    if not str(args.get("near", "") or "").strip():
        return (
            "Tell me an NYC address or landmark and I'll check for scheduled street closures near "
            "it (optionally for a specific date, like a game day)."
        )
    try:
        query = StreetClosureQuery.model_validate(args)
    except ValidationError:
        return (
            "The requested date or result count is invalid. Ask for a date in YYYY-MM-DD format "
            "and a positive result count."
        )
    near = query.near.strip()

    on = (query.visit_date or _nyc_today()).isoformat()

    # PII: geocode the location only to bound the query; the resolved address is never logged.
    origin = await resolve_location(near, ctx)
    if origin is None:
        return f"I could not locate '{near}'. Give me a specific NYC address or landmark."
    if origin.low_confidence:
        return (
            f"'{near}' could match several places. Give me a specific NYC address or landmark and "
            "I'll check street closures there."
        )

    # Filter on the dataset's OWN columns: geometry proximity + the closure's active window.
    where = (
        f"within_circle(the_geom, {origin.lat}, {origin.lon}, {_RADIUS_M}) "
        f"AND work_start_date <= '{on}T23:59:59' AND work_end_date >= '{on}T00:00:00'"
    )
    result = await query_dataset_pages(
        DATASET_ID,
        where=where,
        order="work_start_date",
        page_size=_PAGE_SIZE,
        client=ctx.http,
        _query=query_dataset,
    )
    rows = result.records

    if not rows:
        return (
            f"I found no scheduled DOT street closures within about half a mile of {origin.label} "
            f"on {on}. {_ROUTE}"
        )

    max_results = query.max_results or 5

    lines = [
        f"Scheduled DOT street closures within about half a mile of {origin.label} on {on} "
        f"({len(rows)} found):",
    ]
    for index, record in enumerate(rows[:max_results], 1):
        lines.append(_closure_line(index, record, _closure_citation(ctx, record)))
    if not result.complete:
        lines.append(
            f"The city source stopped after {len(rows)} records, so that count is a lower bound."
        )
    lines.append(_ROUTE)
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_street_closures",
            description=(
                "Check scheduled NYC street closures near a place, grounded in DOT's construction "
                "street-closure schedule (NYC Open Data). Use it for game-day and travel planning: "
                "'are streets closed near <place>?', 'is <street> closed for construction?'. Pass "
                "`near` (one NYC address or landmark) and optional `visit_date` (defaults to today) "
                "to see closures active that day, each with its closed segment, the work "
                "purpose, the scheduled window, and a citation. This is PLANNED construction only: "
                "it does not cover a crash, a parade, a police closure, alternate-side-parking "
                "suspensions, or live traffic. Read-only."
            ),
            input_type=StreetClosureQuery,
            handler=_street_closures,
            open_world=True,  # hits live Socrata
            title="Check NYC street closures",
        )
    ]
