"""Small deterministic time primitives shared by service and event adapters."""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .citations import data_provenance
from .tools.base import Tool, ToolContext

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


class EventIntervalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_name: str = Field(
        min_length=1,
        description="Event name exactly as stated by the cited source.",
    )
    event_date: date | None = Field(
        default=None,
        description="Source-backed event date in New York, when stated.",
    )
    start_time: time | None = Field(
        default=None,
        description="Source-backed local start or check-in time, when stated.",
    )
    end_date: date | None = Field(
        default=None,
        description="Source-backed local end date; omit when it is the event date.",
    )
    end_time: time | None = Field(
        default=None,
        description="Source-backed local end time, when stated.",
    )
    citation_id: str = Field(
        description="Citation ID whose retrieved evidence states these date and time values.",
    )

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def source_time(cls, value):
        if not isinstance(value, str):
            return value
        normalized = value.strip().replace(".", "").upper()
        for fmt in ("%I:%M %p", "%I %p", "%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(normalized, fmt).time()
            except ValueError:
                continue
        return value


class EventIntervalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["success", "citation_not_found", "source_mismatch"] = "success"
    source_citation_id: str
    status_citation_id: str | None = None
    start_at: datetime | None
    end_at: datetime | None
    evaluated_at: datetime
    status: EventStatus


def nyc_datetime(day: date, wall_time: time) -> datetime | None:
    """Attach NYC time only when the source's wall time identifies one instant."""
    value = datetime.combine(day, wall_time, tzinfo=NYC_TZ)
    return value if value.utcoffset() == value.replace(fold=1).utcoffset() else None


def _source_supports(query: EventIntervalQuery, citation: dict) -> bool:
    text = " ".join(
        str(citation.get(field) or "") for field in ("title", "snippet")
    ).lower().replace(".", "")
    text = " ".join(text.split())

    def supports_date(value: date) -> bool:
        fetched_at = ((citation.get("provenance") or {}).get("acquisition") or {}).get(
            "fetched_at"
        )
        if "today" in text and isinstance(fetched_at, str):
            try:
                if datetime.fromisoformat(fetched_at).astimezone(NYC_TZ).date() == value:
                    return True
            except ValueError:
                pass
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
        if str(value.year) not in years:
            return False
        iso = re.search(rf"(?<!\d){re.escape(value.isoformat())}(?!\d)", text)
        month_day = re.search(
            rf"\b(?:{value.strftime('%B')}|{value.strftime('%b')})\s+0?{value.day}(?!\d)",
            text,
            re.IGNORECASE,
        )
        return bool(iso or month_day)

    def supports_time(value: time) -> bool:
        hour = value.hour % 12 or 12
        suffix = "am" if value.hour < 12 else "pm"
        return any(candidate in text for candidate in (
            value.strftime("%H:%M"), value.strftime("%H:%M:%S"),
            f"{hour}:{value.minute:02d} {suffix}",
        ))

    return (
        query.event_name.lower() in text
        and (query.event_date is None or supports_date(query.event_date))
        and (query.end_date is None or supports_date(query.end_date))
        and (query.start_time is None or supports_time(query.start_time))
        and (query.end_time is None or supports_time(query.end_time))
    )


def event_status(
    start_at: datetime | None,
    end_at: datetime | None,
    observed_at: datetime,
) -> EventStatus:
    """Classify a source-backed event interval at one observed instant."""
    if start_at is None:
        return "unknown"
    if end_at is not None and end_at < start_at:
        return "unknown"
    if start_at > observed_at:
        return "upcoming"
    if end_at is None:
        return "unknown"
    return "in_progress" if observed_at < end_at else "ended"


def evaluate_event_interval(
    *,
    event_name: str,
    event_date: date | str | None,
    start_time: time | str | None,
    end_date: date | str | None,
    end_time: time | str | None,
    citation_id: str,
    observed_at: datetime,
) -> EventIntervalEvaluation:
    query = EventIntervalQuery.model_validate({
        "event_name": event_name,
        "event_date": event_date,
        "start_time": start_time,
        "end_date": end_date,
        "end_time": end_time,
        "citation_id": citation_id,
    })
    start_at = (
        nyc_datetime(query.event_date, query.start_time)
        if query.event_date is not None and query.start_time is not None
        else None
    )
    end_at = None
    if query.event_date is not None and query.end_time is not None:
        resolved_end_date = query.end_date or query.event_date
        if (
            query.end_date is None
            and query.start_time is not None
            and query.end_time <= query.start_time
        ):
            resolved_end_date += timedelta(days=1)
        end_at = nyc_datetime(resolved_end_date, query.end_time)
    evaluated_at = observed_at.astimezone(NYC_TZ)
    return EventIntervalEvaluation(
        source_citation_id=query.citation_id,
        start_at=start_at,
        end_at=end_at,
        evaluated_at=evaluated_at,
        status=event_status(start_at, end_at, evaluated_at),
    )


def temporal_tools() -> list[Tool]:
    async def evaluate(args: dict, ctx: ToolContext) -> EventIntervalEvaluation:
        query = EventIntervalQuery.model_validate(args)
        citation = ctx.citations.mapping().get(query.citation_id)
        if citation is None:
            return EventIntervalEvaluation(
                outcome="citation_not_found",
                source_citation_id=query.citation_id,
                start_at=None,
                end_at=None,
                evaluated_at=datetime.now(NYC_TZ),
                status="unknown",
            )
        if not _source_supports(query, citation):
            return EventIntervalEvaluation(
                outcome="source_mismatch",
                source_citation_id=query.citation_id,
                start_at=None,
                end_at=None,
                evaluated_at=datetime.now(NYC_TZ),
                status="unknown",
            )
        result = evaluate_event_interval(
            **query.model_dump(),
            observed_at=datetime.now(NYC_TZ),
        )
        interval = query.model_dump(mode="json")
        computed_citation_id = ctx.citations.register(
            citation["url"],
            title=f"Evaluated event time: {citation.get('title') or 'retrieved event'}",
            snippet=(
                f"Event name: {interval['event_name']}; event date: "
                f"{interval.get('event_date') or 'unknown'}; start time: "
                f"{interval.get('start_time') or 'unknown'}; end date: "
                f"{interval.get('end_date') or interval.get('event_date') or 'unknown'}; "
                f"end time: {interval.get('end_time') or 'unknown'}; evaluated at: "
                f"{result.evaluated_at.isoformat()}; Computed event status: {result.status}."
            ),
            kind="DATA",
            valid_as_of=result.evaluated_at.isoformat(),
            provenance=data_provenance(
                {"source": citation, "interval": interval},
                record_id=f"{query.citation_id}:event-interval:{result.evaluated_at.isoformat()}",
                field_pointer="/interval",
                derivation={
                    "source_citation_id": query.citation_id,
                    "event_name": query.event_name,
                    "start_at": result.start_at.isoformat() if result.start_at else None,
                    "end_at": result.end_at.isoformat() if result.end_at else None,
                    "evaluated_at": result.evaluated_at.isoformat(),
                    "status": result.status,
                },
            ),
        )
        return result.model_copy(update={"status_citation_id": computed_citation_id})

    return [Tool(
        name="evaluate_event_time",
        description=(
            "Deterministically classify a retrieved NYC event as upcoming, in progress, ended, "
            "or unknown. Use this for every web event retained in an answer. Copy its exact event "
            "name, date, available start or end times, and source citation ID. The "
            "tool rejects date or time values absent from that citation. The result returns the "
            "raw New York timestamps, evaluation time, computed status, the input "
            "source_citation_id, and a status_citation_id for that derived result. Cite the "
            "returned status_citation_id for the computed status."
        ),
        parameters=EventIntervalQuery.model_json_schema(),
        handler=evaluate,
        return_type=EventIntervalEvaluation,
        title="Evaluate NYC event time",
    )]
