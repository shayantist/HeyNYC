"""Unit tests for the Tier-2 NLI faithfulness checker (heynyc.core.nli) and its hook in
heynyc.core.grounding.check_grounding.

Every test here is fully offline: MockNLI drives the Tier-2 branch deterministically, and PromptedNLI
is exercised against an INJECTED fake completion. No test loads a real model or touches the network.
The real MiniCheck backend is exercised only by scripts/demo_tier2.py (opt-in, needs the [nli] extra).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from heynyc.core import nli
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


def _fake_completion(content, capture: dict):
    """A stand-in for litellm.completion: records what it was called with and returns an
    OpenAI-shaped response object carrying `content`."""

    def fn(model, messages, **kwargs):
        capture["calls"] = capture.get("calls", 0) + 1
        capture["model"] = model
        capture["messages"] = messages
        capture["kwargs"] = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return fn


def _fake_async_completion(content, capture: dict):
    async def fn(model, messages, **kwargs):
        capture["calls"] = capture.get("calls", 0) + 1
        capture["kwargs"] = kwargs
        usage = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )

    return fn


def test_prompted_nli_builds_prompt_and_parses_unsupported():
    capture: dict = {}
    fake = _fake_completion(
        '{"verdicts":[{"id":"claim-0","label":"unsupported",'
        '"reason":"source never mentions Ley Local 56"}]}',
        capture,
    )
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
    assert "partial" in system.lower() and "contradicted" in system.lower()
    assert capture["kwargs"]["response_format"].__name__ == "NLIBatchResponse"


def test_prompted_nli_marks_clarifying_questions_as_questions() -> None:
    capture: dict = {}
    checker = PromptedNLI(
        completion_fn=_fake_completion(
            '{"verdicts":[{"id":"follow-up","label":"supported",'
            '"reason":"a neutral clarification question"}]}',
            capture,
        )
    )

    checker.check_many([
        nli.NLIInput(
            id="follow-up",
            claim="Did you receive a notice?",
            source="Resident message: Will my benefits stop?",
            kind="question",
        )
    ])

    user = capture["messages"][1]["content"]
    system = capture["messages"][0]["content"]
    assert '"kind": "question"' in user
    assert "need not be entailed" in system
    assert "unsupported premise" in system
    assert "data-minimization reminder" in system
    assert "other factual or procedural advice" in system


def test_prompted_nli_marks_conversational_framing_as_not_always_grounded() -> None:
    capture: dict = {}
    checker = PromptedNLI(
        completion_fn=_fake_completion(
            '{"verdicts":[{"id":"ack","label":"supported",'
            '"reason":"non-factual empathy and uncertainty"}]}',
            capture,
        )
    )

    checker.check_many([
        nli.NLIInput(
            id="ack",
            claim="I understand why you are worried. I cannot tell from this message alone.",
            source="Resident message: Will my benefits stop?",
            kind="framing",
        )
    ])

    user = capture["messages"][1]["content"]
    system = capture["messages"][0]["content"]
    assert '"kind": "framing"' in user
    assert "doesn't require grounding" in system
    assert "external factual or procedural claim" in system


def test_prompted_nli_preserves_leading_question_for_fail_closed_review() -> None:
    capture: dict = {}
    checker = PromptedNLI(
        completion_fn=_fake_completion(
            '{"verdicts":[{"id":"leading","label":"unsupported",'
            '"reason":"the question assumes a work-rule cause"}]}',
            capture,
        )
    )

    verdict = checker.check_many([
        nli.NLIInput(
            id="leading",
            claim="Did your SNAP stop because of the work rule?",
            source="Resident message: Will my benefits stop?",
            kind="question",
        )
    ])[0]

    assert '"kind": "question"' in capture["messages"][1]["content"]
    assert "because of the work rule" in capture["messages"][1]["content"]
    assert verdict.supported is False


def test_prompted_nli_parses_supported_true():
    capture: dict = {}
    fake = _fake_completion(
        '{"verdicts":[{"id":"claim-0","label":"supported",'
        '"reason":"the page says businesses must accept cash"}]}',
        capture,
    )
    verdict = PromptedNLI(completion_fn=fake).check("Businesses must accept cash.", _DCWP_TEXT)
    assert verdict.supported is True
    assert verdict.score >= 0.5


def test_prompted_nli_rejects_prose_wrapped_json():
    capture: dict = {}
    fake = _fake_completion(
        'Sure!\n{"verdicts":[{"id":"claim-0","label":"supported","reason":"stated"}]}\nDone.',
        capture,
    )
    verdict = PromptedNLI(completion_fn=fake).check("Some claim.", "Some source.")
    assert verdict.supported is False


def test_prompted_nli_rejects_non_boolean_legacy_value():
    content = '{"supported":"false","reason":"not stated"}'
    verdict = PromptedNLI(completion_fn=_fake_completion(content, {})).check(
        "Some claim.", "Some source."
    )
    assert verdict.supported is False


def test_prompted_nli_checks_many_in_one_request():
    capture: dict = {}
    content = (
        '{"verdicts":['
        '{"id":"first","label":"supported","reason":"stated"},'
        '{"id":"second","label":"partial","reason":"scope is broader"}'
        "]}"
    )
    verdicts = PromptedNLI(completion_fn=_fake_completion(content, capture)).check_many([
        nli.NLIInput(id="first", claim="Claim one.", source="Source one."),
        nli.NLIInput(id="second", claim="Claim two.", source="Source two."),
    ])

    assert [verdict.supported for verdict in verdicts] == [True, False]
    assert len(capture["messages"]) == 2
    assert capture["calls"] == 1


def test_prompted_nli_fails_closed_when_batch_is_incomplete():
    content = (
        '{"verdicts":['
        '{"id":"first","label":"supported","reason":"stated"}'
        "]}"
    )
    verdicts = PromptedNLI(completion_fn=_fake_completion(content, {})).check_many([
        nli.NLIInput(id="first", claim="Claim one.", source="Source one."),
        nli.NLIInput(id="second", claim="Claim two.", source="Source two."),
    ])

    assert [verdict.supported for verdict in verdicts] == [False, False]
    assert all("incomplete" in verdict.reason for verdict in verdicts)


def test_prompted_nli_fails_safe_on_unparseable_response():
    """A garbled / non-JSON response must NOT be read as a false all-clear: fail safe to UNSUPPORTED
    so the high-recall-of-unsupported-claims posture holds."""
    verdict = PromptedNLI(completion_fn=_fake_completion("I think it is probably fine, yes.", {})).check(
        "Some claim.", "Some source."
    )
    assert verdict.supported is False


def test_prompted_nli_fails_safe_on_non_string_content():
    verdict = PromptedNLI(completion_fn=_fake_completion({"unexpected": "object"}, {})).check(
        "Some claim.", "Some source."
    )
    assert verdict.supported is False


def test_prompted_nli_fails_safe_on_malformed_response_envelope():
    def malformed_completion(model, messages, **kwargs):
        return SimpleNamespace(choices=[])

    verdict = PromptedNLI(completion_fn=malformed_completion).check("Some claim.", "Some source.")
    assert verdict.supported is False


def test_prompted_nli_fails_closed_on_provider_error():
    def failing_completion(model, messages, **kwargs):
        raise RuntimeError("provider unavailable")

    verdict = PromptedNLI(completion_fn=failing_completion).check(
        "Some claim.",
        "Some source.",
    )

    assert verdict.supported is False
    assert verdict.reason == "semantic verifier unavailable"


async def test_prompted_nli_async_batch_returns_usage_and_one_request():
    capture: dict = {}
    content = (
        '{"verdicts":['
        '{"id":"first","label":"supported","reason":"stated"},'
        '{"id":"second","label":"unsupported","reason":"not stated"}'
        "]}"
    )
    run = await PromptedNLI(
        model="openai/gpt-5.4-nano",
        async_completion_fn=_fake_async_completion(content, capture),
    ).arun_many([
        nli.NLIInput(id="first", claim="Claim one.", source="Source one."),
        nli.NLIInput(id="second", claim="Claim two.", source="Source two."),
    ])

    assert isinstance(run, nli.NLIBatchRun)
    assert [verdict.supported for verdict in run.verdicts] == [True, False]
    assert run.input_tokens == 120
    assert run.output_tokens == 30
    assert run.cached_input_tokens == 20
    assert run.cost_usd is not None
    assert run.latency_ms >= 0
    assert capture["calls"] == 1
    assert capture["kwargs"]["response_format"].__name__ == "NLIBatchResponse"


async def test_prompted_nli_default_transport_omits_temperature(monkeypatch):
    capture: dict = {}
    content = (
        '{"verdicts":['
        '{"id":"first","label":"supported","reason":"stated"}'
        "]}"
    )

    async def completion(**kwargs):
        capture.update(kwargs)
        return await _fake_async_completion(content, {})(
            kwargs["model"],
            kwargs["messages"],
            response_format=kwargs["response_format"],
        )

    monkeypatch.setattr("litellm.acompletion", completion)

    run = await PromptedNLI(model="openai/gpt-5.6-luna").arun_many([
        nli.NLIInput(id="first", claim="Claim one.", source="Source one."),
    ])

    assert run.verdicts[0].supported is True
    assert "temperature" not in capture


async def test_prompted_nli_async_batch_fails_closed_on_provider_error():
    async def failing_completion(model, messages, **kwargs):
        raise RuntimeError("provider unavailable")

    run = await PromptedNLI(
        async_completion_fn=failing_completion,
    ).arun_many([
        nli.NLIInput(id="first", claim="Claim one.", source="Source one."),
        nli.NLIInput(id="second", claim="Claim two.", source="Source two."),
    ])

    assert [verdict.supported for verdict in run.verdicts] == [False, False]
    assert run.error == "RuntimeError"
    assert run.latency_ms >= 0
