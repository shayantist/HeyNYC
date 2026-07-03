from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from heynyc.core.prompts import build_system_prompt
from heynyc.core.registry import Registry


def test_system_prompt_injects_current_nyc_datetime():
    fixed = datetime(2026, 6, 28, 19, 30, tzinfo=ZoneInfo("America/New_York"))
    prompt = build_system_prompt(Registry([]), now=fixed)
    assert "Current date & time" in prompt
    assert "June 28, 2026" in prompt
    assert "America/New_York" in prompt
    # still carries the grounding rules
    assert "GROUND EVERYTHING" in prompt


def test_system_prompt_includes_active_recency_check():
    # The freshness guard goes from passive date-stamping to an ACTIVE recency check: on
    # time-sensitive law/policy/rights questions the agent must run recent_developments and
    # surface any breaking change as a dated, cited heads-up on top of the official answer.
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "recent_developments" in prompt
    assert "this may be changing" in low
    assert "recency check" in low


def test_system_prompt_surfaces_human_and_appeal_path():
    # When it can't help or a user reports a denial/problem, the agent must offer a way to
    # reach a human (311/the agency) and the official appeal path. (NYC GenAI Guidance.)
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "311" in prompt
    assert "appeal" in low
    assert "human" in low
