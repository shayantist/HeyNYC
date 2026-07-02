"""Capability discovery — data-driven from each module's `examples:` (multichannel spec §14)."""
from __future__ import annotations

from heynyc.core import config
from heynyc.core.registry import Registry


def _reg():
    return Registry.discover(config.MODULES_DIR)


def test_modules_carry_examples():
    benefits = next(m for m in _reg().modules if m.name == "benefits")
    assert len(benefits.examples) >= 2
    assert all(isinstance(e, str) and e for e in benefits.examples)


def test_welcome_examples_are_spread_and_capped():
    ex = _reg().welcome_examples(4)
    assert 1 <= len(ex) <= 4
    assert all(e for e in ex)
    assert len(set(ex)) == len(ex)        # no duplicates


def test_capability_menu_rows_are_category_blurb_examples():
    menu = _reg().capability_menu()
    assert menu, "expected at least the benefits module"
    assert "benefits" in {cat for cat, _blurb, _ex in menu}
    assert all(isinstance(ex, list) for _cat, _blurb, ex in menu)


def test_welcome_text_leads_with_examples_never_how_can_i_help():
    text = _reg().welcome_text()
    assert "How can I help" not in text          # the anti-MyCity / "show, don't ask" rule
    assert "•" in text                            # concrete example bullets
    assert "SNAP" in text or "groceries" in text  # real, groundable capabilities


def test_welcome_text_discloses_it_is_an_ai_not_a_city_employee():
    # EU AI Act Art 50 + NYC GenAI transparency: first contact must disclose it's an AI, not staff.
    low = _reg().welcome_text().lower()
    assert "ai assistant" in low
    assert "not a city employee" in low


def test_is_help_detects_greetings_but_not_real_questions():
    from heynyc.channels.orchestrator import is_help
    assert is_help("hi") and is_help("Help!") and is_help("what can you do?")
    assert is_help("/menu") and is_help("  hello  ")
    assert not is_help("where's the nearest cooling center?")   # a real question is NOT help
    assert not is_help("am i eligible for snap")
