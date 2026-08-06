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
    "I'm really sorry you're in this much pain. Call 911 right now. Call 988 now too. "
    "Move away from anything you could use to hurt yourself and contact someone you trust who "
    "can stay with you. I'm an AI and can't call or monitor emergency help for you."
)
_EN_IDEATION = (
    "I'm really sorry you're hurting. I can't help with ways to end your life. "
    "Call 988 right now. If you might act now or cannot stay safe, call 911. "
    "Move away from anything you could use to hurt yourself and contact someone you trust."
)

_OFFICIAL_HOSTS = (
    "nyc.gov",
    "cityofnewyork.us",
    "988lifeline.org",
    "omh.ny.gov",
)


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


# F149: ht/ko/ur/pl were recorded as having no official translation, and an Urdu resident got an
# all-English crisis reply because of it. NYS OMH publishes this page in all four at
# `<lang>.omh.ny.gov`, the same host the Bengali record already used
@pytest.mark.parametrize("code", ["ht", "ko", "ur", "pl"])
def test_omh_harvested_language_carries_verbatim_official_copy(code):
    line = CRISIS_LINES[code]

    assert line.has_verified_copy
    assert "988" in line.lifeline_988
    assert line.source_988 == f"https://{code}.omh.ny.gov/omhweb/crisis/what-is-988.html"
    assert line.verified_on == "2026-07-30"
    # The page's 911 text is an explanatory comparison, not an instruction, so the English floor
    # supplies 911; the note must say so rather than implying an in-language 911 route exists
    assert not line.emergency_911
    assert "911" in line.note


def test_every_ll30_language_now_carries_verified_copy():
    missing = [code for code, line in CRISIS_LINES.items() if not line.has_verified_copy]

    assert not missing, f"no verified crisis copy for {missing}"


def test_bengali_uses_current_official_988_and_911_copy():
    line = CRISIS_LINES["bn"]

    assert line.lifeline_988
    assert line.emergency_911
    assert line.source_988 == "https://bn.omh.ny.gov/omhweb/crisis/what-is-988.html"
    assert line.source_911 == "https://access.nyc.gov/bn/programs/nyc-988/"


def test_bengali_floor_does_not_promise_in_language_text_support():
    response = compose_crisis_floor(_EN_IMMINENT, "bn")

    assert "text 988" not in response
    assert "ডায়াল করুন 988" in response


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


# Arabic is the one LL30 language whose official page carries a clean 911 line but no clean 988
# line, so it gets the cited interpretation fact rather than an in-language 988 pointer (F149)
def test_language_without_a_988_pointer_states_the_interpretation_fact():
    out = compose_crisis_floor(_EN_IDEATION, "ar")

    assert out.startswith(_EN_IDEATION)
    assert "240 languages" in out
    assert "no cost" in out
    assert "988" in out and "911" in out


@pytest.mark.parametrize("code", ["ht", "ko", "ur", "pl"])
def test_omh_language_gets_its_own_copy_not_the_english_interpreter_line(code):
    out = compose_crisis_floor(_EN_IDEATION, code)

    assert CRISIS_LINES[code].lifeline_988 in out
    assert "240 languages" not in out


def test_unknown_language_stays_byte_identical():
    # No established language means no interpretation promise to make
    assert compose_crisis_floor(_EN_IDEATION, None) == _EN_IDEATION
    assert compose_crisis_floor(_EN_IDEATION, "de") == _EN_IDEATION  # unknown code


@pytest.mark.parametrize("code", ["zh", "bn", "es", "ru", "fr"])
def test_language_with_verified_copy_does_not_get_the_english_interpreter_line(code):
    # Inverse: a resident whose language has verified official copy gets that, not an English
    # sentence about interpreters
    assert "240 languages" not in compose_crisis_floor(_EN_IDEATION, code)


def test_every_language_floor_carries_988_and_911_exactly():
    # The hard invariant: whatever language routing yields, the served floor always carries both
    # numbers (the English floor supplies them even when no verified translation exists).
    for code in LL30_LANGUAGES:
        out = compose_crisis_floor(_EN_IDEATION, code)
        assert "988" in out, f"{code} floor lost 988"
        assert "911" in out, f"{code} floor lost 911"
