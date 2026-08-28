from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core.temporal import (
    interval_status,
    nyc_datetime,
    parse_clock_minutes,
    weekly_open_status,
)


def test_event_status_uses_source_interval_and_nyc_clock():
    nyc = ZoneInfo("America/New_York")
    observed_at = datetime(2026, 8, 20, 14, 24, tzinfo=nyc)

    assert interval_status(
        datetime(2026, 8, 20, 5, 15, tzinfo=nyc),
        datetime(2026, 8, 20, 9, 30, tzinfo=nyc),
        observed_at,
    ) == "ended"
    assert interval_status(
        datetime(2026, 8, 20, 14, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 15, 0, tzinfo=nyc),
        observed_at,
    ) == "in_progress"
    assert interval_status(
        datetime(2026, 8, 20, 19, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 21, 0, tzinfo=nyc),
        observed_at,
    ) == "upcoming"


def test_event_status_is_unknown_when_the_source_does_not_bound_the_event():
    nyc = ZoneInfo("America/New_York")
    observed_at = datetime(2026, 8, 20, 14, 24, tzinfo=nyc)

    assert interval_status(None, None, observed_at) == "unknown"
    assert interval_status(
        datetime(2026, 8, 20, 5, 15, tzinfo=nyc), None, observed_at,
    ) == "unknown"
    assert interval_status(
        datetime(2026, 8, 20, 20, 0, tzinfo=nyc),
        datetime(2026, 8, 20, 9, 30, tzinfo=nyc),
        observed_at,
    ) == "unknown"


def test_event_status_handles_an_overnight_interval():
    nyc = ZoneInfo("America/New_York")

    assert interval_status(
        datetime(2026, 8, 20, 23, 0, tzinfo=nyc),
        datetime(2026, 8, 21, 2, 0, tzinfo=nyc),
        datetime(2026, 8, 21, 1, 0, tzinfo=nyc),
    ) == "in_progress"


def test_event_status_is_unknown_for_an_ambiguous_or_missing_nyc_wall_time():
    ambiguous = nyc_datetime(datetime(2026, 11, 1).date(), datetime.strptime("01:30", "%H:%M").time())
    missing = nyc_datetime(datetime(2026, 3, 8).date(), datetime.strptime("02:30", "%H:%M").time())

    assert ambiguous is None
    assert missing is None


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


def test_interval_status_uses_raw_source_backed_datetimes():
    nyc = ZoneInfo("America/New_York")
    start_at = datetime(2026, 8, 20, 5, 15, tzinfo=nyc)
    end_at = datetime(2026, 8, 20, 9, 30, tzinfo=nyc)
    observed_at = datetime(2026, 8, 20, 14, 24, tzinfo=nyc)

    assert interval_status(start_at, end_at, observed_at) == "ended"
