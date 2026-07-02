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


def test_system_prompt_surfaces_human_and_appeal_path():
    # When it can't help or a user reports a denial/problem, the agent must offer a way to
    # reach a human (311/the agency) and the official appeal path. (NYC GenAI Guidance.)
    prompt = build_system_prompt(Registry([]))
    low = prompt.lower()
    assert "311" in prompt
    assert "appeal" in low
    assert "human" in low
