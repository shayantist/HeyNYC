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

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset
from heynyc.core.tools.geo import geocode, maps_link

# ponytail: single source for the dataset id; the manifest declares the matching
# binding (category street_closure) only so the capability table / grounding label read it.
DATASET_ID = "i6b5-j7bu"
_NYC_TZ = ZoneInfo("America/New_York")
_RADIUS_M = 800   # ~0.5 mile, the bound handed to Socrata within_circle
_MAX_ROWS = 200   # cap the query; the geo+date filter already narrows hard

_ROUTE = (
    "This is DOT's planned construction-closure schedule, so it will not show a sudden crash, a "
    "parade, or a police closure. For live traffic and incidents check 511NY (511ny.org) or "
    "nyc.gov/dot, and dial 511."
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


def _parse_on_date(raw: str, today: date) -> str:
    """Sanitize the asked date to an ISO YYYY-MM-DD string, defaulting to today.

    A value that is not a real ISO date (including any SoQL-injection attempt) can never
    reach the query: it falls back to today. The returned value is always digits-and-dashes,
    so interpolating it into the $where clause is safe."""
    raw = (raw or "").strip()
    if raw:
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            pass
    return today.isoformat()


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


async def _street_closures(args: dict, ctx: ToolContext) -> str:
    near = str(args.get("near", "") or "").strip()
    if not near:
        return (
            "Tell me an NYC address or landmark and I'll check for scheduled street closures near "
            "it (optionally for a specific date, like a game day)."
        )

    on = _parse_on_date(str(args.get("on", "") or ""), _nyc_today())

    # PII: geocode the location only to bound the query; the resolved address is never logged.
    origin = await geocode(near, client=ctx.http)
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
    rows = await query_dataset(
        DATASET_ID, where=where, order="work_start_date", limit=_MAX_ROWS, client=ctx.http
    )

    if not rows:
        return (
            f"I found no scheduled DOT street closures within about half a mile of {origin.label} "
            f"on {on}. {_ROUTE}"
        )

    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 10)

    lines = [
        f"Scheduled DOT street closures within about half a mile of {origin.label} on {on} "
        f"({len(rows)} found):",
    ]
    for index, record in enumerate(rows[:limit], 1):
        lines.append(_closure_line(index, record, _closure_citation(ctx, record)))
    lines.append(_ROUTE)
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="street_closures",
            description=(
                "Check scheduled NYC street closures near a place, grounded in DOT's construction "
                "street-closure schedule (NYC Open Data). Use it for game-day and travel planning: "
                "'are streets closed near <place>?', 'is <street> closed for construction?'. Pass "
                "near (one NYC address or landmark) and optional on (a date, YYYY-MM-DD; defaults "
                "to today) to see closures active that day, each with its closed segment, the work "
                "purpose, the scheduled window, and a citation. This is PLANNED construction only: "
                "it does not cover a crash, a parade, a police closure, alternate-side-parking "
                "suspensions, or live traffic. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {
                        "type": "string",
                        "description": "One NYC address or landmark to check closures around.",
                    },
                    "on": {
                        "type": "string",
                        "description": "Date to check (YYYY-MM-DD), e.g. a game day. Defaults to today.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Max closures to list.",
                    },
                },
            },
            handler=_street_closures,
            open_world=True,  # hits live Socrata
            title="Check NYC street closures",
        )
    ]
