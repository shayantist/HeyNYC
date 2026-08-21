from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.temporal import (
    evaluate_event_interval,
    event_status,
    parse_clock_minutes,
    temporal_tools,
    weekly_open_status,
)
from heynyc.core.tools.base import ToolContext


def test_event_status_uses_source_interval_and_nyc_clock():
    nyc = ZoneInfo("America/New_York")
    observed_at = datetime(2026, 8, 20, 14, 24, tzinfo=nyc)

    assert event_status(
        datetime(2026, 8, 20, 5, 15, tzinfo=nyc),
        datetime(2026, 8, 20, 9, 30, tzinfo=nyc),
        observed_at,
    ) == "ended"
    assert event_status(
        datetime(2026, 8, 20, 14, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 15, 0, tzinfo=nyc),
        observed_at,
    ) == "in_progress"
    assert event_status(
        datetime(2026, 8, 20, 19, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 21, 0, tzinfo=nyc),
        observed_at,
    ) == "upcoming"


def test_event_status_is_unknown_when_the_source_does_not_bound_the_event():
    nyc = ZoneInfo("America/New_York")
    observed_at = datetime(2026, 8, 20, 14, 24, tzinfo=nyc)

    assert event_status(None, None, observed_at) == "unknown"
    assert event_status(
        datetime(2026, 8, 20, 5, 15, tzinfo=nyc), None, observed_at,
    ) == "unknown"
    assert event_status(
        datetime(2026, 8, 20, 20, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 9, 30, tzinfo=nyc),
        observed_at,
    ) == "unknown"


def test_event_status_handles_an_overnight_interval():
    nyc = ZoneInfo("America/New_York")

    assert event_status(
        datetime(2026, 8, 20, 23, 0, tzinfo=nyc),
        datetime(2026, 8, 21, 2, 0, tzinfo=nyc),
        datetime(2026, 8, 21, 1, 0, tzinfo=nyc),
    ) == "in_progress"


def test_event_status_is_unknown_for_an_ambiguous_or_missing_nyc_wall_time():
    nyc = ZoneInfo("America/New_York")

    ambiguous = evaluate_event_interval(
        event_name="DST concert",
        event_date="2026-11-01",
        start_time="01:30:00",
        end_date=None,
        end_time="02:30:00",
        citation_id="S1",
        observed_at=datetime(2026, 11, 1, 3, 0, tzinfo=nyc),
    )
    missing = evaluate_event_interval(
        event_name="DST concert",
        event_date="2026-03-08",
        start_time="02:30:00",
        end_date=None,
        end_time="04:00:00",
        citation_id="S2",
        observed_at=datetime(2026, 3, 8, 4, 30, tzinfo=nyc),
    )

    assert ambiguous.status == "unknown"
    assert ambiguous.start_at is None
    assert missing.status == "unknown"
    assert missing.start_at is None


def test_shared_weekly_schedule_math_handles_formats_unknown_hours_and_overnight_windows():
    nyc = ZoneInfo("America/New_York")
    schedule = {
        0: [(parse_clock_minutes("9:00 AM"), parse_clock_minutes("17:00"))],
        6: [(parse_clock_minutes("2200"), parse_clock_minutes("2:00 AM"))],
    }

    assert weekly_open_status(schedule, datetime(2026, 8, 17, 10, 0, tzinfo=nyc)) is True
    assert weekly_open_status(schedule, datetime(2026, 8, 17, 18, 0, tzinfo=nyc)) is False
    assert weekly_open_status(schedule, datetime(2026, 8, 16, 23, 0, tzinfo=nyc)) is True
    assert weekly_open_status(schedule, datetime(2026, 8, 17, 1, 0, tzinfo=nyc)) is True
    assert weekly_open_status({}, datetime(2026, 8, 17, 10, 0, tzinfo=nyc)) is None


async def test_temporal_tool_returns_raw_interval_and_computed_literal():
    nyc = ZoneInfo("America/New_York")
    result = evaluate_event_interval(
        event_name="RAYE Live on TODAY",
        event_date="2026-08-20",
        start_time="05:15:00",
        end_date=None,
        end_time="09:30:00",
        citation_id="S1",
        observed_at=datetime(2026, 8, 20, 14, 24, tzinfo=nyc),
    )

    assert result.status == "ended"
    assert result.start_at.isoformat() == "2026-08-20T05:15:00-04:00"
    assert result.end_at.isoformat() == "2026-08-20T09:30:00-04:00"
    assert result.evaluated_at.isoformat() == "2026-08-20T14:24:00-04:00"
    assert result.source_citation_id == "S1"
    assert result.status_citation_id is None

    tool = temporal_tools()[0]
    assert tool.name == "evaluate_event_time"
    assert tool.return_type is not None


async def test_temporal_tool_rejects_a_citation_not_retrieved_this_turn():
    tool = temporal_tools()[0]
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await tool.handler(
        {
            "event_name": "RAYE Live on TODAY",
            "event_date": "2026-08-20",
            "start_time": "05:15:00",
            "end_time": "09:30:00",
            "citation_id": "S99",
        },
        ctx,
    )

    assert result.outcome == "citation_not_found"
    assert result.status == "unknown"
    assert result.source_citation_id == "S99"
    assert result.status_citation_id is None


async def test_temporal_tool_accepts_natural_source_times_and_rejects_mismatched_values():
    tool = temporal_tools()[0]
    citations = CitationRegistry()
    cite = citations.register(
        "https://example.com/event",
        snippet="RAYE Live. Date: August 20, 2026. Check-in by 5:15 a.m. Concert concludes at 9:30 a.m.",
    )
    ctx = ToolContext(citations=citations, registry=Registry([]))

    accepted = await tool.handler(
        {
            "event_name": "RAYE Live",
            "event_date": "2026-08-20",
            "start_time": "5:15 a.m.",
            "end_time": "9:30 AM",
            "citation_id": cite,
        },
        ctx,
    )
    rejected = await tool.handler(
        {
            "event_name": "RAYE Live",
            "event_date": "2026-08-20",
            "start_time": "7:00 PM",
            "end_time": "9:00 PM",
            "citation_id": cite,
        },
        ctx,
    )
    wrong_year = await tool.handler(
        {
            "event_name": "RAYE Live",
            "event_date": "2027-08-20",
            "start_time": "5:15 a.m.",
            "end_time": "9:30 AM",
            "citation_id": cite,
        },
        ctx,
    )
    wrong_day = await tool.handler(
        {
            "event_name": "RAYE Live",
            "event_date": "2026-08-02",
            "start_time": "5:15 a.m.",
            "end_time": "9:30 AM",
            "citation_id": cite,
        },
        ctx,
    )
    wrong_event = await tool.handler(
        {
            "event_name": "Different concert",
            "event_date": "2026-08-20",
            "start_time": "5:15 a.m.",
            "end_time": "9:30 AM",
            "citation_id": cite,
        },
        ctx,
    )
    yearless = citations.register(
        "https://example.com/yearless-event",
        snippet="RAYE Live. Date: August 20. Check-in by 5:15 a.m. Concert concludes at 9:30 a.m.",
    )
    unsupported_year = await tool.handler(
        {
            "event_name": "RAYE Live",
            "event_date": "2027-08-20",
            "start_time": "5:15 a.m.",
            "end_time": "9:30 AM",
            "citation_id": yearless,
        },
        ctx,
    )
    today_cite = citations.register(
        "https://example.com/today-event",
        snippet="MindTravel Piano Concert Today at 7:00 PM",
        provenance={"acquisition": {"fetched_at": "2026-08-20T16:55:50-04:00"}},
    )
    relative_today = await tool.handler(
        {
            "event_name": "MindTravel Piano Concert",
            "event_date": "2026-08-20",
            "start_time": "7:00 PM",
            "citation_id": today_cite,
        },
        ctx,
    )

    assert accepted.outcome == "success"
    assert accepted.start_at.hour == 5
    assert accepted.end_at.hour == 9
    assert accepted.source_citation_id == cite
    assert accepted.status_citation_id != cite
    derived = citations.mapping()[accepted.status_citation_id]
    assert derived["kind"] == "DATA"
    assert derived["provenance"]["derivation"]["source_citation_id"] == cite
    assert f"Computed event status: {accepted.status}" in derived["snippet"]
    assert rejected.outcome == "source_mismatch"
    assert rejected.status == "unknown"
    assert wrong_year.outcome == "source_mismatch"
    assert wrong_day.outcome == "source_mismatch"
    assert wrong_event.outcome == "source_mismatch"
    assert unsupported_year.outcome == "source_mismatch"
    assert relative_today.outcome == "success"
