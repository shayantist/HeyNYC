"""Unit tests for the Tier-2 NLI faithfulness checker (heynyc.core.nli) and its hook in
heynyc.core.grounding.check_grounding.

Every test here is fully offline: MockNLI drives the Tier-2 branch deterministically, and PromptedNLI
is exercised against an INJECTED fake completion. No test loads a real model or touches the network.
The real MiniCheck backend is exercised only by scripts/demo_tier2.py (opt-in, needs the [nli] extra).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from heynyc.core.grounding import NLIMismatch, check_grounding
from heynyc.core.nli import MockNLI, NLIVerdict, PromptedNLI

# Real captured text of the DCWP cashless page (a WEB citation), fetched 2026-07-11 from
# https://www.nyc.gov/site/dca/consumers/Prohibition-of-Cashless-Establishments.page — it SUPPORTS
# "businesses must accept cash" but contains NO "Local Law 56" and no "2021". This is the exact gap
# Tier-1 is silent on: a fabricated statute cited to a soft (WEB) source.
_DCWP_TEXT = (
    "Prohibition of Cashless Establishments. NYC businesses must accept cash unless they have a "
    "machine to convert cash to a prepaid card. They cannot charge more for paying in cash. You can "
    "file a complaint about a retail or food store, including a food cart, in New York City that "
    "refuses cash payments. Cash means U.S. currency and coins. Your store may refuse cash payments "
    "for telephone, mail, or internet-based transactions, unless the transaction takes place in the "
    "store. Read Guidance for Food Stores and Retail Establishments Regarding Telephone and "
    "Internet-based Transaction Exceptions to the Cashless Establishment Ban."
)

# The fabrication: a Spanish sentence that conflates a SUPPORTED clause ("must accept cash") with an
# INVENTED statute ("Ley Local 56 de 2021") the source never states. Cited to the WEB page above.
_FABRICATION = (
    "Bajo la Ley Local 56 de 2021, los restaurantes deben aceptar efectivo. {cite:S3}"
)
# The clean answer: names the real law DESCRIPTIVELY ("the Cashless Ban Law"), no invented number.
_CLEAN = "Under the Cashless Ban Law, food establishments must accept cash. {cite:S3}"


def _web_cite(snippet: str = _DCWP_TEXT) -> dict:
    """A WEB citation: snippet + title only, no provenance snapshot — so Tier-1 treats it as an
    EXCERPT (complete == 0) and never blocks on it. This is the class Tier-2 is here to cover."""
    return {
        "url": "https://www.nyc.gov/site/dca/consumers/Prohibition-of-Cashless-Establishments.page",
        "kind": "WEB",
        "title": "Prohibition of Cashless Establishments - DCWP",
        "snippet": snippet,
    }


def _statute_rule(claim: str, source: str) -> bool:
    """Deterministic stand-in for the model: a claim that asserts a specific 'Ley Local NN' statute is
    UNSUPPORTED unless that exact statute string appears in the source. Mirrors what a faithfulness
    model does on the fixture (the invented statute is not in the DCWP text) without any inference."""
    m = re.search(r"ley local \d+", claim.lower())
    return not (m and m.group(0) not in source.lower())


# --- Tier-2 hook in check_grounding, driven by MockNLI --------------------------------------------


def test_mock_nli_flags_fabricated_statute_cited_to_web_source():
    res = check_grounding(_FABRICATION, {"S3": _web_cite()}, nli=MockNLI(_statute_rule))
    assert res is not None
    assert res.nli_checked == 1
    assert len(res.nli_failures) == 1, res.nli_failures
    fail = res.nli_failures[0]
    assert isinstance(fail, NLIMismatch)
    assert fail.cited == ["S3"]
    assert "Ley Local 56" in fail.claim
    assert "{cite:S3}" not in fail.claim  # the marker is stripped before the model sees the claim
    assert fail.score < 0.5


def test_mock_nli_passes_clean_descriptive_claim():
    res = check_grounding(_CLEAN, {"S3": _web_cite()}, nli=MockNLI(_statute_rule))
    assert res is not None
    assert res.nli_checked == 1
    assert res.nli_failures == []


def test_tier1_alone_stays_silent_on_the_fabrication():
    """The whole reason Tier-2 exists: with no checker, Tier-1 does NOT block the fabricated statute
    (WEB excerpt + proper-noun mismatch is soft), so the lie ships. Tier-2 is what catches it."""
    res = check_grounding(_FABRICATION, {"S3": _web_cite()})
    assert res is None or (res.blocking is False and not res.hard_failures)


def test_nli_failure_makes_passed_false_but_not_blocking_by_default():
    res = check_grounding(_FABRICATION, {"S3": _web_cite()}, nli=MockNLI(_statute_rule))
    assert res is not None
    assert res.passed is False           # the answer is not fully grounded
    assert res.blocking is False         # but Tier-2 does not block unless nli_blocking=True


def test_nli_blocking_flag_lets_tier2_block():
    res = check_grounding(
        _FABRICATION, {"S3": _web_cite()}, nli=MockNLI(_statute_rule), nli_blocking=True
    )
    assert res is not None
    assert res.nli_failures
    assert res.blocking is True


# --- MockNLI construction modes -------------------------------------------------------------------


def test_mock_nli_mapping_mode_keys_on_normalized_claim():
    nli = MockNLI({"a supported claim": True, "an unsupported claim": False})
    assert nli.check("A Supported Claim", "src").supported is True
    v = nli.check("An  Unsupported   Claim", "src")  # case + whitespace normalized to the key
    assert v.supported is False and v.score < 0.5


def test_mock_nli_float_rule_applies_threshold():
    v = MockNLI(lambda claim, source: 0.3).check("x", "y")
    assert isinstance(v, NLIVerdict)
    assert v.supported is False and v.score == 0.3 and v.backend == "mock"


# --- PromptedNLI: prompt build + response parse, against an injected fake completion ---------------


def _fake_completion(content: str, capture: dict):
    """A stand-in for litellm.completion: records what it was called with and returns an
    OpenAI-shaped response object carrying `content`."""

    def fn(model, messages):
        capture["model"] = model
        capture["messages"] = messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return fn


def test_prompted_nli_builds_prompt_and_parses_unsupported():
    capture: dict = {}
    fake = _fake_completion('{"supported": false, "reason": "source never mentions Ley Local 56"}', capture)
    nli = PromptedNLI(model="ollama/qwen3.5:9b", completion_fn=fake)
    verdict = nli.check("Bajo la Ley Local 56 de 2021, deben aceptar efectivo.", _DCWP_TEXT)

    assert verdict.backend == "prompted"
    assert verdict.supported is False
    assert verdict.score < 0.5
    assert "Ley Local 56" in verdict.reason

    # Prompt build: claim and source both reach the model, in the user turn.
    user = "\n".join(m["content"] for m in capture["messages"] if m["role"] == "user")
    assert "Ley Local 56" in user
    assert "must accept cash" in user
    # Judge support, not truth: the instruction must say so.
    system = "\n".join(m["content"] for m in capture["messages"] if m["role"] == "system")
    assert "support" in system.lower()
    assert "true" in system.lower() and "false" in system.lower()  # JSON-only contract


def test_prompted_nli_parses_supported_true():
    capture: dict = {}
    fake = _fake_completion('{"supported": true, "reason": "the page says businesses must accept cash"}', capture)
    verdict = PromptedNLI(completion_fn=fake).check("Businesses must accept cash.", _DCWP_TEXT)
    assert verdict.supported is True
    assert verdict.score >= 0.5


def test_prompted_nli_tolerates_prose_wrapped_json():
    capture: dict = {}
    fake = _fake_completion('Sure! Here is my verdict:\n{"supported": false, "reason": "not stated"}\nHope that helps.', capture)
    verdict = PromptedNLI(completion_fn=fake).check("Some claim.", "Some source.")
    assert verdict.supported is False


def test_prompted_nli_fails_safe_on_unparseable_response():
    """A garbled / non-JSON response must NOT be read as a false all-clear: fail safe to UNSUPPORTED
    so the high-recall-of-unsupported-claims posture holds."""
    verdict = PromptedNLI(completion_fn=_fake_completion("I think it is probably fine, yes.", {})).check(
        "Some claim.", "Some source."
    )
    assert verdict.supported is False
