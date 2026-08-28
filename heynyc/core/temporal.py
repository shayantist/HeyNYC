"""Small deterministic time primitives shared by service and event adapters."""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

EventStatus = Literal["upcoming", "in_progress", "ended", "unknown"]
NYC_TZ = ZoneInfo("America/New_York")


def parse_clock_minutes(value: object) -> int | None:
    """Parse common source clock formats into minutes after midnight."""
    text = str(value or "").strip().replace(".", "").upper()
    if not text or text == "NULL":
        return None
    if text.isdigit() and len(text) in {3, 4}:
        hour, minute = divmod(int(text), 100)
        return hour * 60 + minute if hour < 24 and minute < 60 else None
    for fmt in ("%I:%M %p", "%I %p", "%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.hour * 60 + parsed.minute
    return None


def weekly_open_status(
    intervals: dict[int, list[tuple[int, int]]], observed_at: datetime,
) -> bool | None:
    """Evaluate recurring weekly intervals, including windows crossing midnight."""
    known = any(slots for slots in intervals.values())
    if not known:
        return None
    minute = observed_at.hour * 60 + observed_at.minute
    today = intervals.get(observed_at.weekday(), ())
    if any(
        (opened < closed and opened <= minute < closed)
        or (opened >= closed and minute >= opened)
        for opened, closed in today
    ):
        return True
    previous = intervals.get((observed_at.weekday() - 1) % 7, ())
    if any(opened >= closed and minute < closed for opened, closed in previous):
        return True
    return False


def nyc_datetime(day: date, wall_time: time) -> datetime | None:
    """Attach NYC time only when the source's wall time identifies one instant."""
    value = datetime.combine(day, wall_time, tzinfo=NYC_TZ)
    return value if value.utcoffset() == value.replace(fold=1).utcoffset() else None


def interval_status(
    start_at: datetime | None,
    end_at: datetime | None,
    observed_at: datetime,
) -> EventStatus:
    """Classify any source-backed interval at one observed instant."""
    if start_at is None:
        return "unknown"
    if end_at is not None and end_at < start_at:
        return "unknown"
    if start_at > observed_at:
        return "upcoming"
    if end_at is None:
        return "unknown"
    return "in_progress" if observed_at < end_at else "ended"
