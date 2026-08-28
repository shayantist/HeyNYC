"""Current MTA subway elevator and escalator outages."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated

import httpx
from pydantic import Field

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext, ToolFailure, ToolInput

_OUTAGES_URL = (
    "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/"
    "nyct%2Fnyct_ene.json"
)
_STATUS_PAGE = "https://www.mta.info/elevator-escalator-status"


class ElevatorStatusInput(ToolInput):
    stations: list[Annotated[str, Field(description="MTA subway station")]] = Field(
        min_length=1,
        description="MTA subway stations",
    )


def _station_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _matches(requested: str, station: str, lines: str) -> bool:
    route_suffix = re.fullmatch(
        r"(.*?)[-(]\s*((?:[a-z0-9]+\s*/\s*)+[a-z0-9]+)\)?",
        requested.casefold(),
    )
    station_name = route_suffix.group(1) if route_suffix else requested
    wanted = _station_key(station_name)
    actual = _station_key(station)
    if not wanted or not (wanted in actual or actual in wanted):
        return False
    if route_suffix is None:
        return True
    requested_lines = set(re.findall(r"[a-z0-9]+", route_suffix.group(2)))
    actual_lines = set(re.findall(r"[a-z0-9]+", lines.casefold()))
    return requested_lines <= actual_lines


async def _mta_elevator_status(
    args: ElevatorStatusInput,
    ctx: ToolContext,
) -> str | ToolFailure:
    stations = list(dict.fromkeys(
        str(value).strip()
        for value in args.get("stations", [])
        if str(value).strip()
    ))
    if not stations:
        return "Give one or more MTA station names to check."

    own_client = ctx.http is None
    client = ctx.http or httpx.AsyncClient(timeout=20.0)
    try:
        response = await client.get(_OUTAGES_URL)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("unexpected MTA outage response")
    except (httpx.HTTPError, ValueError):
        return ToolFailure(
            status="unavailable",
            reason="The current MTA elevator and escalator outage feed could not be read.",
            retryable=True,
            source_url=_STATUS_PAGE,
        )
    finally:
        if own_client:
            await client.aclose()

    checked_at = datetime.now(UTC).isoformat()
    snapshot = {"outages": rows}
    cite = ctx.citations.register(
        _OUTAGES_URL,
        snippet=(
            f"Current MTA elevator and escalator outage feed checked for: "
            f"{', '.join(stations)}"
        ),
        title="MTA elevator and escalator outage feed",
        kind="DATA",
        provenance=data_provenance(
            snapshot,
            record_id="current-outages",
            field_pointer="/outages",
            derivation={
                "requested_stations": stations,
                "checked_at": checked_at,
            },
        ),
    )
    status_cite = ctx.citations.register(
        _STATUS_PAGE,
        snippet="Official MTA elevator and escalator status page",
        title="MTA elevator and escalator status",
        kind="WEB",
    )

    lines = [f"MTA outage feed checked at {checked_at}:"]
    for station in stations:
        matches = [
            row
            for row in rows
            if _matches(
                station,
                str(row.get("station", "")),
                str(row.get("trainno", "")),
            )
        ]
        if not matches:
            lines.append(
                f"- {station}: no matching outage was listed. This is not a guarantee that every "
                f"elevator is working; recheck the MTA status page before leaving. {{cite:{cite}}}"
            )
            continue
        for row in matches:
            equipment = str(row.get("equipment", "")).strip() or "equipment"
            serving = str(row.get("serving", "")).strip() or "location not specified"
            reason = str(row.get("reason", "")).strip() or "reason not listed"
            estimate = str(row.get("estimatedreturntoservice", "")).strip()
            estimate_text = f"; estimated return {estimate}" if estimate else ""
            lines.append(
                f"- {row.get('station', station)}: {equipment}, {serving}; {reason}"
                f"{estimate_text}. {{cite:{cite}}}"
            )
    lines.append(f"Official status page: {_STATUS_PAGE} {{cite:{status_cite}}}")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="check_mta_elevators",
            description=(
                "Check the current MTA subway elevator and escalator outage feed for named "
                "stations. Call this once after identifying the stations an accessible "
                "route depends on. Add slash-separated lines after the station name, such as "
                "'86 St-4/5/6', when multiple stations share a name. A station absent from the "
                "outage feed is not guaranteed to "
                "have every elevator working, so preserve the tool's recheck warning."
            ),
            input_type=ElevatorStatusInput,
            handler=_mta_elevator_status,
            open_world=True,
        )
    ]
