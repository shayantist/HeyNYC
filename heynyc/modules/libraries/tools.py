"""Current Brooklyn Public Library branch locations and today's hours."""
from __future__ import annotations

from html import unescape

import httpx
from pydantic import Field

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import Place
from heynyc.core.tools.geo import miles, rank_nearby, resolve_location

_BPL_LOCATIONS_URL = "https://www.bklynlibrary.org/api/locations/v1/map"


class LibraryQuery(LocationRequest):
    near: str = Field(description="NYC place to search near.")
    max_results: int | None = Field(
        default=None, ge=1, le=8, description="Maximum branches requested; omit for the default 5."
    )


def _position(row: dict) -> tuple[float, float] | None:
    try:
        lat, lon = str(row["position"]).split(",", 1)
        return float(lat), float(lon)
    except (KeyError, TypeError, ValueError):
        return None


def _plain_hours(value: object) -> str:
    return " ".join(unescape(str(value).rpartition(">")[2]).split())


def _branch(row: dict) -> Place | None:
    point = _position(row)
    title = str(row.get("title", "")).strip()
    if point is None or not title:
        return None
    return Place(
        name=title,
        lat=point[0],
        lon=point[1],
        address=str(row.get("address", "")).strip(),
        phone=str(row.get("phone", "")).strip(),
        website=str(row.get("path", "")).strip(),
        hours=_plain_hours(row.get("hours", "")).removeprefix("Today's Hours: "),
        record_id=str(row.get("branchid", title)),
        raw=row,
    )


async def _find_bpl_branches(args: dict, ctx: ToolContext) -> str:
    query = LibraryQuery.model_validate(args)
    origin = await resolve_location(query.near, ctx)
    if origin is None or origin.low_confidence:
        return f"Could not confidently locate '{query.near}' in NYC. Ask for a nearby address or cross street."

    own_client = ctx.http is None
    client = ctx.http or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(_BPL_LOCATIONS_URL)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("unexpected BPL locations response")
    except (httpx.HTTPError, ValueError):
        return "Could not verify current Brooklyn Public Library branch information from its live feed."
    finally:
        if own_client:
            await client.aclose()

    max_results = query.max_results or 5
    branches = [branch for row in payload if (branch := _branch(row)) is not None]
    ranked = rank_nearby(
        origin,
        branches,
        key=lambda branch: branch.record_id or branch.name.casefold(),
        limit=max_results,
    )
    lines = [f"Nearest BPL branches to {origin.label} from the current official branch feed:"]
    for branch, distance_m in ranked:
        row = branch.raw
        distance_miles = miles(distance_m)
        title = branch.name
        address = branch.address or "address not listed"
        phone = branch.phone or "phone not listed"
        hours = branch.hours
        hours = f"today's listed hours: {hours}" if hours else "today's hours not listed"
        closure = _plain_hours(row.get("closingmsg", ""))
        has_notice = bool(closure)
        branch_url = branch.website or _BPL_LOCATIONS_URL
        cite = ctx.citations.register(
            branch_url,
            snippet=f"{title}; {address}; {phone}; {hours}; {closure}",
            title=title,
            kind="DATA",
            provenance=data_provenance(
                row,
                record_id=str(row.get("branchid", title)),
                field_pointer="/",
                derivation={
                    "origin": [origin.lat, origin.lon],
                    "point": [branch.lat, branch.lon],
                    "distance_mi": distance_miles,
                    "source": _BPL_LOCATIONS_URL,
                },
            ),
        )
        lines.append(
            f"- {title}: {distance_miles:.2f} mi straight-line; {address}; {hours}; "
            f"phone {phone}; branch page: {branch_url}"
            f"{'; listed hours unconfirmed because the feed also says: ' + closure if has_notice else ''} {{cite:{cite}}}"
        )
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_bpl_branches",
            description=(
                "Find Brooklyn Public Library branches nearest an NYC place using BPL's current "
                "official branch feed. Returns today's listed hours, address, phone, "
                "feed notice, branch page, and computed straight-line distance. Use for BPL branch location or "
                "current-hours questions. Results stay in distance order. Treat listed hours as "
                "unconfirmed whenever the same row also has a notice. The feed does not prove that an unlisted service is "
                "unavailable; use the official branch page or web search for extra details."
            ),
            parameters=LibraryQuery.model_json_schema(),
            handler=_find_bpl_branches,
            open_world=True,
            title="Find BPL branches",
        )
    ]
