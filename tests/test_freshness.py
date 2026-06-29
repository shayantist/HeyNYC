from __future__ import annotations

from heynyc.core.freshness import days_old, staleness_caveat


def test_days_old_basic():
    assert days_old("2026-06-01", "2026-06-29") == 28
    assert days_old("2026-06-02T00:00:00.000", "2026-06-29") == 27  # ISO datetime tolerated
    assert days_old("nonsense", "2026-06-29") is None
    assert days_old("", "2026-06-29") is None


def test_staleness_caveat_fires_only_when_too_old():
    # within tolerance → no caveat
    assert staleness_caveat("2026-06-02", "2026-06-29", max_days=365) == ""
    # a year-old benefits figure → caveat
    caveat = staleness_caveat("2024-01-01", "2026-06-29", max_days=365)
    assert caveat.startswith("⚠️")
    assert "over 2 years old" in caveat
    assert "2024-01-01" in caveat


def test_staleness_caveat_lenient_on_unknown_and_future():
    assert staleness_caveat("", "2026-06-29", max_days=365) == ""        # unknown → no double-warn
    assert staleness_caveat("2027-01-01", "2026-06-29", max_days=365) == ""  # future (clock skew) → fresh
