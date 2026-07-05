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


def test_system_prompt_refuses_obfuscated_encoded_instructions():
    # Red-team PI12 fix: an encoded (base64/hex/rot13) "do what it says" instruction must be refused,
    # not executed, and never answered with silence.
    low = build_system_prompt(Registry([])).lower()
    assert "base64" in low
    assert "encoded" in low or "obfuscated" in low
    assert "non-empty" in low or "never fall silent" in low


def test_system_prompt_forbids_uncited_authority_on_substantive_facts():
    # Grounding-discipline fix: an ungrounded substantive claim must not be framed as authoritative;
    # the agent grounds it via a tool or gives it as general guidance routed to 311/official — while
    # still correcting harmful misconceptions (never over-abstaining).
    low = build_system_prompt(Registry([])).lower()
    assert "authoritative answer" in low            # the anti-pattern it must avoid
    assert "general information" in low             # the allowed ungrounded framing
    assert "misconception" in low                   # must still correct false premises
    assert "emtali" in low or "emtala" in low       # names the find_clinic grounding route


def test_system_prompt_describes_data_practices_accurately():
    # Privacy-accuracy fix: the prompt must NOT let the agent claim it "stores nothing"; it should
    # describe the in-progress application draft honestly and match the PIA.
    low = build_system_prompt(Registry([])).lower()
    assert "draft" in low
    assert "profile" in low                          # "don't build a profile"
    assert "store nothing" in low or "retain nothing" in low  # named as the claim to AVOID


def test_system_prompt_sets_plain_language_reading_level():
    low = build_system_prompt(Registry([])).lower()
    assert "reading level" in low
    assert "6th" in low and "8th" in low


def test_system_prompt_contested_legal_matter_protocol():
    # Red-team MC03/MC04/FP02/ES03 fix: on a contested legal ruling surfaced from news, the agent must
    # NOT restate the court/holding/scope, must NOT tell a user their protection is struck down/gone/
    # annulled, and must LEAD with the protection that currently stands, then route.
    low = build_system_prompt(Registry([])).lower()
    assert "contested legal matter" in low
    # names the exact words it must never use about a still-valid right
    assert "struck down" in low and "annulled" in low
    # the safe shape: lead with what currently stands
    assert "currently stands" in low
    assert "active legal challenge" in low
    # must not assert the court
    assert "never name the court" in low or "which court ruled" in low


def test_system_prompt_emergency_no_medical_dosing():
    # Red-team ES09 fix: an emergency (heart-attack signs) gets 911 only — NO drug names or dosages,
    # not even aspirin. The English path already held; this makes the standard explicit for all langs.
    low = build_system_prompt(Registry([])).lower()
    assert "911" in low
    assert "no medical instructions" in low or "no dosages" in low
    assert "aspirin" in low  # named as the exact thing NOT to dose


def test_system_prompt_public_charge_snap_guardrail():
    # Red-team FP03 fix: the agent must NOT say SNAP counts toward public charge; SNAP is generally
    # NOT a public-charge benefit, and it must not derive the rule from news about "the administration."
    low = build_system_prompt(Registry([])).lower()
    assert "public charge" in low
    assert "snap" in low and "not counted" in low
    assert "actionnyc" in low  # routes to immigration legal aid


def test_system_prompt_same_discipline_in_every_language():
    # Red-team ES09/ES13 fix: EVERY safety + grounding rule holds identically in Spanish and any other
    # language (emergencies -> 911 no dosing; never fabricate a law number/citation; lead with the
    # standing protection). A prompt reinforcement only partially closes a model-level cross-lingual gap.
    low = build_system_prompt(Registry([])).lower()
    assert "every language" in low
    assert "law number" in low  # never invent a law number in any language
    # cites the concrete correct statute the Spanish ES13 failure fabricated ("Local Law 68")
    assert "local law 34" in low or "20-840" in low
