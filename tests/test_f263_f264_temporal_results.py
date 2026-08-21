from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.ticketmaster import TicketmasterSearchResult
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.cooling_centers import tools as cooling
from heynyc.modules.events import tools as events


def _context(query: str) -> ToolContext:
    return ToolContext(citations=CitationRegistry(), registry=Registry([]), query=query)


def _cooling_row(
    record_id: int,
    name: str,
    lat: float,
    *,
    saturday: tuple[str, str] | None = None,
) -> dict:
    row = {
        "OBJECTID": record_id,
        "NYCEM_ID": str(record_id),
        "Facility_name": name,
        "Address": f"{record_id} Main St",
        "lat": lat,
        "lon": -73.978,
        "Finder_status": "OPEN",
        "Space_type": "Cooling Center",
        "Age_restriction": "No",
    }
    if saturday:
        row.update(
            {
                "Saturday": f"{saturday[0]}-{saturday[1]}",
                "cc_sat_open1": saturday[0],
                "cc_sat_close1": saturday[1],
            }
        )
    return row


def _patch_cooling_lookup(monkeypatch, rows: list[dict]) -> None:
    async def fake_geocode(text, **kwargs):
        return GeoPoint(40.758, -73.978, text)

    async def fake_query(url, **kwargs):
        return rows

    monkeypatch.setattr(cooling, "geocode", fake_geocode)
    monkeypatch.setattr(cooling, "query_feature_service", fake_query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 8, 15, 20, 30, tzinfo=ZoneInfo("America/New_York")),
    )


@pytest.mark.asyncio
async def test_f263_explicit_same_day_date_is_planning_not_open_now(monkeypatch) -> None:
    _patch_cooling_lookup(
        monkeypatch,
        [_cooling_row(1, "Saturday Library", 40.7581, saturday=("09:00 AM", "05:00 PM"))],
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "visit_date": "2026-08-15"},
        _context("Where can we cool down in Flushing on Saturday?"),
    )

    assert "for Saturday, August 15, 2026" in output
    assert "Scheduled Saturday, August 15, 2026: 09:00 AM-05:00 PM" in output
    assert "closed right now" not in output
    assert "scheduled closed now" not in output


@pytest.mark.asyncio
async def test_f263_agent_extracted_weekday_date_reaches_deterministic_schedule(monkeypatch) -> None:
    _patch_cooling_lookup(
        monkeypatch,
        [_cooling_row(1, "Saturday Library", 40.7581, saturday=("09:00 AM", "05:00 PM"))],
    )

    output = await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "visit_date": "2026-08-15"},
        _context("Free events and a cooling place near Flushing on Saturday"),
    )

    assert "for Saturday, August 15, 2026" in output
    assert "Scheduled Saturday, August 15, 2026: 09:00 AM-05:00 PM" in output
    assert "scheduled closed now" not in output


@pytest.mark.asyncio
async def test_f263_tool_computes_planned_local_time_and_keeps_unknown_distinct(monkeypatch) -> None:
    closed = _cooling_row(3, "Closed Library", 40.7583)
    closed["Saturday"] = "CLOSED"
    _patch_cooling_lookup(
        monkeypatch,
        [
            _cooling_row(1, "Open Library", 40.7581, saturday=("09:00 AM", "05:00 PM")),
            _cooling_row(2, "Unknown Library", 40.7582),
            closed,
        ],
    )

    output = await cooling.get_tools()[0].handler(
        {
            "near": "Flushing, Queens",
            "visit_date": "2026-08-15",
            "visit_time": "12:00",
        },
        _context("What will be open Saturday at noon near Flushing, Queens?"),
    )

    assert "scheduled open at 12:00 PM America/New_York" in output
    assert "schedule unknown at 12:00 PM America/New_York" in output
    assert "scheduled closed at 12:00 PM America/New_York" in output

    schema = cooling.get_tools()[0].parameters["properties"]
    assert {item.get("format") for item in schema["visit_date"]["anyOf"]} == {
        "date",
        None,
    }
    assert {item.get("format") for item in schema["visit_time"]["anyOf"]} == {
        "time",
        None,
    }


@pytest.mark.asyncio
async def test_f263_current_aggregate_has_result_set_citation(monkeypatch) -> None:
    _patch_cooling_lookup(
        monkeypatch,
        [
            _cooling_row(1, "Closed Nearby", 40.7581, saturday=("09:00 AM", "05:00 PM")),
            _cooling_row(2, "Open Farther", 40.768, saturday=("07:00 PM", "11:00 PM")),
        ],
    )

    ctx = _context("What is open right now near Flushing?")
    output = await cooling.get_tools()[0].handler(
        {"near": "Flushing, Queens", "open_now_only": True},
        ctx,
    )

    aggregate = next(line for line in output.splitlines() if "closer option is" in line)
    assert "{cite:S" in aggregate
    citation_id = aggregate.split("{cite:", 1)[1].split("}", 1)[0]
    derivation = ctx.citations.mapping()[citation_id]["provenance"]["derivation"]
    assert derivation["origin"] == [40.758, -73.978]
    assert derivation["predicate"] == "open_now is false and distance_m < nearest_open_distance_m"
    assert derivation["nearest_open"]["record_id"] == "2"
    assert derivation["distances_m"] == [
        {"record_id": "1", "value": pytest.approx(11.12, abs=0.01)}
    ]


def test_f263_planned_status_handles_overnight_hours() -> None:
    record = {"cc_sat_open1": "10:00 PM", "cc_sat_close1": "02:00 AM"}
    nyc = ZoneInfo("America/New_York")

    assert cooling._open_now(record, datetime(2026, 8, 15, 23, 0, tzinfo=nyc)) is True
    assert cooling._open_now(record, datetime(2026, 8, 16, 1, 0, tzinfo=nyc)) is True
    assert cooling._open_now(record, datetime(2026, 8, 16, 2, 0, tzinfo=nyc)) is False


@pytest.mark.asyncio
async def test_f264_event_tool_defaults_to_five_and_exposes_max_results(monkeypatch) -> None:
    async def fake_ticketmaster(**kwargs):
        return TicketmasterSearchResult(
            status="complete",
            events=[
                {
                    "id": str(index),
                    "name": f"Event {index}",
                    "url": f"https://example.com/{index}",
                    "dates": {"start": {"localDate": "2099-08-15", "localTime": "12:00:00"}},
                }
                for index in range(6)
            ],
            retrieved_at="2099-08-01T00:00:00+00:00",
        )

    async def empty_dataset(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", empty_dataset)

    output = await events.get_tools()[0].handler(
        {"window_start": "2099-08-15", "window_end": "2099-08-15"},
        _context("events on August 15"),
    )

    assert output.count("(Ticketmaster Discovery)") == 5
    schema = events.get_tools()[0].parameters["properties"]["max_results"]
    assert schema["default"] is None
    integer_schema = next(item for item in schema["anyOf"] if item.get("type") == "integer")
    assert integer_schema["maximum"] == 20


@pytest.mark.asyncio
async def test_f264_agent_extracted_max_results_controls_the_shortlist(monkeypatch) -> None:
    async def fake_ticketmaster(**kwargs):
        return TicketmasterSearchResult(
            status="complete",
            events=[
                {
                    "id": str(index),
                    "name": f"Event {index}",
                    "url": f"https://example.com/{index}",
                    "dates": {
                        "start": {"localDate": "2099-08-15", "localTime": "12:00:00"}
                    },
                }
                for index in range(12)
            ],
            retrieved_at="2099-08-01T00:00:00+00:00",
        )

    async def empty_dataset(*args, **kwargs):
        return []

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", empty_dataset)

    output = await events.get_tools()[0].handler(
        {
            "window_start": "2099-08-15",
            "window_end": "2099-08-15",
            "max_results": 10,
        },
        _context("please give me ten events on August 15"),
    )

    assert output.count("(Ticketmaster Discovery)") == 10


@pytest.mark.asyncio
async def test_f264_web_and_catalog_choices_share_the_same_limit(monkeypatch) -> None:
    async def fake_ticketmaster(**kwargs):
        return TicketmasterSearchResult(
            status="complete",
            events=[
                {
                    "id": str(index),
                    "name": f"Event {index}",
                    "dates": {"start": {"localDate": "2099-08-15"}},
                }
                for index in range(6)
            ],
        )

    async def empty_dataset(*args, **kwargs):
        return []

    async def web_handler(args, ctx):
        citation_id = ctx.citations.register(
            "https://example.org/seasonal-event",
            title="One current seasonal lead",
            snippet="One current seasonal event on August 15.",
        )
        return f"[{citation_id}] One current seasonal lead"

    monkeypatch.setattr(events, "ticketmaster_events", fake_ticketmaster)
    monkeypatch.setattr(events, "query_dataset", empty_dataset)
    ctx = _context("events on August 15")
    ctx.toolbox = {
        "web_search": Tool(
            name="web_search",
            description="search",
            parameters={"type": "object"},
            handler=web_handler,
        )
    }

    output = await events.get_tools()[0].handler(
        {"window_start": "2099-08-15", "window_end": "2099-08-15"}, ctx,
    )

    assert output.count("(Ticketmaster Discovery)") == 4
    assert output.count("(Web discovery;") == 1
    assert output.count("\n- ") == 5
