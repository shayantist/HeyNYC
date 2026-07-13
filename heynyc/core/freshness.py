"""Source freshness, the active half of the §12 staleness guard.

Provenance (the per-source `valid_as_of` date) already rides on every Citation. This adds the
*guard*: when a source is older than a module's tolerance, the tool emits a caveat so the agent
flags it instead of presenting a possibly-stale figure (an income limit, a deadline) as current.

The as-of date is always the source's `updated_at`, never fetch/cache time, caching must never
make stale data look fresh. Tolerances are per data type: benefits rules re-check annually,
real-time status in hours/days, events are handled by future-dating instead.
"""
from __future__ import annotations

from datetime import date
from typing import Optional


def days_old(as_of: str, today: str) -> Optional[int]:
    """Whole days between an 'as of' date and today (both 'YYYY-MM-DD'). None if unparseable."""
    try:
        a = date.fromisoformat((as_of or "")[:10])
        t = date.fromisoformat((today or "")[:10])
    except ValueError:
        return None
    return (t - a).days


def staleness_caveat(as_of: str, today: str, max_days: int) -> str:
    """A short caveat when the source is older than `max_days`, else ''.

    An unknown/unparseable date returns '', the citation already surfaces 'as of unknown',
    so we don't double-warn. A future as-of date (clock skew) is treated as fresh."""
    age = days_old(as_of, today)
    if age is None or age <= max_days:
        return ""
    years = age // 365
    span = f"over {years} year{'s' if years != 1 else ''}" if years >= 1 else f"about {age} days"
    return (
        f"⚠️ this is {span} old (as of {as_of[:10]}), figures like income limits and "
        "deadlines change, so confirm the current details at the official source before relying on it"
    )
