"""Read-only 311 service-request status lane, grounded in NYC Open Data erm2-nwe9.

Two resident intents, one tool, no keyword routing: the model picks arguments by
meaning and the handler branches on which structured argument is present.

  - "is my complaint moving?"  -> sr_number: look up one row by its own unique_key
    (the SR number). ONLY the number is sent; a resident's address is never geocoded
    or logged on this path.
  - "what's happening with 311 complaints about X near me?" -> about / near: recent
    rows filtered on the dataset's own columns (complaint_type, created_date) and the
    location point column (within_circle), summarized by status.

Every fact is cited with a full-row DATA provenance snapshot and a re-fetchable
permalink keyed on the row's own unique_key; valid_as_of comes from the record's own
dates. Filing (Create-SR) is out of scope: this lane cannot change a 311 request.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset
from heynyc.core.tools.geo import geocode, maps_link

# ponytail: single source for the dataset id; the manifest declares the matching binding
# (category service_request_311) only so the capability table / grounding label read it.
DATASET_ID = "erm2-nwe9"
_NYC_TZ = ZoneInfo("America/New_York")
_RECENT_DAYS = 30   # area lane window: "recent" 311 activity, keeps the query cheap + honest
_RADIUS_M = 800     # ~0.5 mile, the "near me" bound handed to Socrata within_circle

_CADENCE_NOTE = (
    "NYC's 311 status data updates about once a day, so a very recent change may not show yet. "
    "To report a new problem or see the official status, use portal.311.nyc.gov or call 311."
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
        lines.append(
            "This request is marked closed. If the problem is still happening, that counts as a new "
            "issue, report it to 311."
        )
    else:
        as_of = _valid_as_of(record)[:10]
        lines.append(
            f"This request is still open (status {status})"
            + (f"; its last recorded update was {as_of}." if as_of else ".")
        )
    lines.append(_CADENCE_NOTE)
    return "\n".join(lines)


async def _lookup_area(about: str, near: str, limit: int, ctx: ToolContext) -> str:
    cutoff = (_nyc_now() - timedelta(days=_RECENT_DAYS)).strftime("%Y-%m-%dT00:00:00")
    clauses = [f"created_date > '{cutoff}'"]
    if about:
        safe = about.replace("'", "''")  # SoQL escapes a quote by doubling it
        clauses.append(f"upper(complaint_type) like upper('%{safe}%')")

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
        clauses.append(f"within_circle(location, {origin.lat}, {origin.lon}, {_RADIUS_M})")

    rows = await query_dataset(
        DATASET_ID,
        where=" AND ".join(clauses),
        order="created_date DESC",
        limit=200,
        client=ctx.http,
    )

    scope = f'about "{about}" ' if about else ""
    where_label = f"near {origin.label}" if origin else "in that area"
    if not rows:
        return (
            f"I found no recent NYC 311 complaints {scope}{where_label} in the last {_RECENT_DAYS} "
            "days. That could mean none were filed, or they are older than this window. To report a "
            "new problem or check official status, use portal.311.nyc.gov or call 311."
        )

    counts = Counter(str(r.get("status", "") or "").strip() or "Unknown" for r in rows)
    breakdown = ", ".join(f"{n} {status}" for status, n in counts.most_common())
    lines = [
        f"Recent NYC 311 complaints {scope}{where_label} (created in the last {_RECENT_DAYS} days):",
        f"- Of the {len(rows)} I found, status is: {breakdown}",
        "Most recent:",
    ]
    for index, record in enumerate(rows[:limit], 1):
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
    lines.append(_CADENCE_NOTE)
    return "\n".join(lines)


async def _nyc311_status(args: dict, ctx: ToolContext) -> str:
    sr_number = str(args.get("sr_number", "") or "").strip()
    if sr_number:  # most specific intent wins; this path never geocodes
        return await _lookup_sr(sr_number, ctx)

    about = str(args.get("about", "") or "").strip()
    near = str(args.get("near", "") or "").strip()
    if not about and not near:
        return (
            "Tell me your 311 service request number to check its status, or a complaint type and an "
            "NYC location (like 'noise complaints near Union Square') to see recent 311 activity."
        )
    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 10)
    return await _lookup_area(about, near, limit, ctx)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="nyc311_status",
            description=(
                "Check an EXISTING NYC 311 service request or recent 311 activity, grounded in the "
                "city's 311 Service Requests data. Pass sr_number to look up one request's status "
                "by its service-request number (send only the number). Or pass about (the problem, "
                "in the resident's words, e.g. 'noise', 'heat', 'illegal parking') and/or near (one "
                "NYC address or landmark) to see recent matching complaints and how they're moving. "
                "Read-only: it reports status, it cannot file, update, or close a 311 request."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sr_number": {
                        "type": "string",
                        "description": "The resident's 311 service request number (digits only).",
                    },
                    "about": {
                        "type": "string",
                        "description": "Complaint topic to match, e.g. 'noise', 'heat', 'illegal parking'.",
                    },
                    "near": {
                        "type": "string",
                        "description": "One NYC address or landmark to bound recent complaints.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Max recent complaints to list (area lookup).",
                    },
                },
            },
            handler=_nyc311_status,
            open_world=True,  # hits live Socrata
            title="Check 311 request status",
        )
    ]
