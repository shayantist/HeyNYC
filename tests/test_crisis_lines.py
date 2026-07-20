"""Crisis-line translations for the Local Law 30 languages (phase 1 of the crisis screen).

The response floor is authored in English and served verbatim; these tests pin the DATA table of
human-verified official crisis copy (988 / 911) per LL30 language, the deterministic language router,
and the composer that appends the verified in-language pointer to the English floor. Nothing here is
machine-translated: every non-English string is copied verbatim from an official government / SAMHSA
crisis page, so the tests assert provenance (official source URL + verification date) as well as text.
"""
from __future__ import annotations

import pytest

from heynyc.core.crisis_lines import (
    CRISIS_LINES,
    LL30_LANGUAGES,
    compose_crisis_floor,
)

# The English self-harm floors the composer wraps (byte-identical to agent.py's constants).
_EN_IMMINENT = (
    "Call 911 right now. Call or text 988 now too. Move away from anything you could use "
    "to hurt yourself and contact someone you trust who can stay with you. I'm an AI and "
    "can't call or monitor emergency help for you."
)
_EN_IDEATION = (
    "I'm really sorry you're hurting. I can't help with ways to end your life. "
    "Call or text 988 right now. If you might act now or cannot stay safe, call 911. "
    "Move away from anything you could use to hurt yourself and contact someone you trust."
)

_OFFICIAL_HOSTS = ("nyc.gov", "cityofnewyork.us", "988lifeline.org")


def test_table_covers_all_ten_ll30_languages():
    assert set(LL30_LANGUAGES) == {"es", "zh", "ru", "bn", "ht", "ko", "ar", "ur", "fr", "pl"}
    assert set(CRISIS_LINES) == set(LL30_LANGUAGES)
    for code, line in CRISIS_LINES.items():
        assert line.lang == code
        assert line.name == LL30_LANGUAGES[code]


def test_verified_copy_carries_the_exact_numbers_and_official_sources():
    for line in CRISIS_LINES.values():
        if line.lifeline_988:
            assert "988" in line.lifeline_988
            assert any(host in line.source_988 for host in _OFFICIAL_HOSTS)
            assert line.verified_on
        if line.emergency_911:
            assert "911" in line.emergency_911
            assert any(host in line.source_911 for host in _OFFICIAL_HOSTS)
            assert line.verified_on


def test_no_verified_copy_language_has_an_honest_note():
    # bn/ht/ko/ur/pl: no official human translation of crisis copy was located; that MUST be stated.
    for code in ("bn", "ht", "ko", "ur", "pl"):
        line = CRISIS_LINES[code]
        assert not line.has_verified_copy
        assert line.note, f"{code} must honestly state that no verified translation exists"


@pytest.mark.parametrize("english_floor", [_EN_IMMINENT, _EN_IDEATION])
def test_covered_language_appends_verified_pointer(english_floor):
    # Chinese has both a verified 988 and a verified 911 line: the composed floor is the English floor
    # plus BOTH official Chinese lines, verbatim.
    out = compose_crisis_floor(english_floor, "zh")
    zh = CRISIS_LINES["zh"]
    assert out.startswith(english_floor)
    assert zh.lifeline_988 in out
    assert zh.emergency_911 in out


def test_language_with_only_988_appends_only_that_line():
    # Russian: only a verified 988 line exists (no verified 911 line); do not fabricate a 911 line.
    out = compose_crisis_floor(_EN_IDEATION, "ru")
    ru = CRISIS_LINES["ru"]
    assert ru.lifeline_988 in out
    assert ru.emergency_911 == ""


def test_language_with_only_911_appends_only_that_line():
    # Arabic: NYC publishes a verified 911 line, but no clean in-language 988 copy was located.
    out = compose_crisis_floor(_EN_IDEATION, "ar")
    ar = CRISIS_LINES["ar"]
    assert ar.emergency_911 in out
    assert ar.lifeline_988 == ""


def test_uncovered_language_falls_back_to_english_byte_identical():
    for code in ("bn", "ht", "ko", "ur", "pl"):
        assert compose_crisis_floor(_EN_IDEATION, code) == _EN_IDEATION
    assert compose_crisis_floor(_EN_IDEATION, None) == _EN_IDEATION
    assert compose_crisis_floor(_EN_IDEATION, "de") == _EN_IDEATION  # unknown code


def test_every_language_floor_carries_988_and_911_exactly():
    # The hard invariant: whatever language routing yields, the served floor always carries both
    # numbers (the English floor supplies them even when no verified translation exists).
    for code in LL30_LANGUAGES:
        out = compose_crisis_floor(_EN_IDEATION, code)
        assert "988" in out, f"{code} floor lost 988"
        assert "911" in out, f"{code} floor lost 911"
