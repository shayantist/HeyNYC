import asyncio
import inspect

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from heynyc.eval.cases import EvalCase
from heynyc.eval.checks import run_checks
from heynyc.eval.invariants import build_invariant_checks
from heynyc.eval.runner import CaseResult
from heynyc.eval.trace import Trace


def test_release_gate_does_not_grade_resident_prose_with_string_matching():
    case = EvalCase(
        id="mechanical-boundary",
        module="global",
        query="help",
        expect_contains=["magic phrase"],
        abstain=True,
        harm_category="injection",
        language="en",
        invariants={
            "must_not_fabricate": True,
            "must_abstain_or_redirect": True,
            "forbid_compliance": True,
            "must_not_claim_current": True,
            "must_offer_immediate_action": True,
        },
    )
    trace = Trace(
        case_id=case.id,
        query=case.query,
        language=case.language,
        final_text="arbitrary resident-facing prose",
    )
    case_result = CaseResult(
        case=case,
        text=("administrative eligibility documentation requirements " * 12).strip(),
    )

    invariant_names = {check.name for check in build_invariant_checks(trace, case)}
    check_names = {
        check.name
        for check in asyncio.run(run_checks(case_result, link_checker=lambda _: None))
    }

    assert invariant_names.isdisjoint(
        {
            "faithfulness",
            "abstain_or_redirect",
            "forbid_compliance",
            "currentness",
            "resident_outcome_floor",
        }
    )
    assert "contains" not in check_names
    assert "readability" not in check_names


@pytest.mark.asyncio
async def test_every_live_case_requires_full_trace_qualitative_review():
    from heynyc.eval.report import evaluate

    case = EvalCase(id="ordinary", module="global", query="what can I do today?")
    report = await evaluate([CaseResult(case=case, text="A completed answer")])

    assert report.reports[0].qualitative_review_required
    assert not report.promotion_ready


def test_public_runtime_redacts_pii_instead_of_semantically_short_circuiting():
    from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter

    source = inspect.getsource(PydanticRuntimeAdapter._run)

    assert "_sensitive_identifier_backstop" not in source
    assert "redact_sensitive_identifiers" in source


def test_channel_orchestrator_does_not_classify_emergencies_from_text():
    from heynyc.channels.orchestrator import handle

    assert "_emergency_backstop" not in inspect.getsource(handle)


@pytest.mark.asyncio
async def test_public_runtime_redacts_sensitive_id_but_keeps_location_context():
    from types import SimpleNamespace

    from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
    from heynyc.core.registry import Registry

    screened = []
    model_messages = []

    async def screen(turns):
        screened.extend(turns)
        return SimpleNamespace(
            risk="none",
            language="en",
            model="test",
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            requests=0,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    async def model(messages, _info):
        model_messages.extend(messages)
        return ModelResponse([TextPart("Use the official secure form")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
        guard_grounding=False,
    ).run("My SSN is 123-45-6789 and I am near 350 Jay Street")

    projected = str(model_messages)
    assert "123-45-6789" not in projected
    assert "123-45-6789" not in " ".join(screened)
    assert "350 Jay Street" in projected
    assert result.diagnostics["pii_redacted"] is True
