from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.nli import NLIBatchRun, NLIVerdict
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import UNVERIFIED_DRAFT_NOTICE
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


class _OwnershipVerifier:
    def __init__(self, *, partial: bool) -> None:
        self.partial = partial
        self.inputs = []

    async def arun_many(self, inputs):
        self.inputs.append(inputs)
        return NLIBatchRun([
            NLIVerdict(
                supported=not self.partial,
                score=0.0 if self.partial else 1.0,
                backend="fake",
                label="partial" if self.partial else "supported",
            )
            for _ in inputs
        ], latency_ms=12.0)


async def _run(
    answer: str,
    verifier: _OwnershipVerifier,
    *,
    grounded: bool = False,
):
    async def source(_args: dict, ctx: ToolContext) -> str:
        food = ctx.citations.register(
            "https://finder.nyc.gov/foodhelp",
            title="Food Help NYC",
            kind="WEB",
            snippet=(
                "Food Help NYC lists places to get food now. "
                "[Open the map](https://foodhelp.nyc.gov/)."
            ),
        )
        review = ctx.citations.register(
            "https://www.nyc.gov/site/hra/about/contact.page",
            title="HRA contact",
            kind="WEB",
            snippet="Contact HRA to ask about a case decision.",
        )
        return f"Food help. {{cite:{food}}} HRA contact. {{cite:{review}}}"

    calls = 0

    async def model(_messages, _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                "grounded_answer" if grounded else "final_answer",
                (
                    {
                        "grounded_blocks": [
                            {"text": answer, "citation_ids": ["S1"]},
                        ]
                    }
                    if grounded
                    else {"answer": answer}
                ),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        semantic_verifier=verifier,
    ).run("How can I get help with food and my HRA case?")
    return result, calls


async def test_misowned_compound_claim_is_preserved_and_labeled_unverified() -> None:
    answer = (
        "HRA can review the closure, and Food Help NYC can help now. {cite:S1}"
    )
    verifier = _OwnershipVerifier(partial=True)

    result, calls = await _run(answer, verifier)

    assert calls == 2
    assert result.text.startswith(
        f"{UNVERIFIED_DRAFT_NOTICE}\n\n"
        "Unverified: HRA can review the closure, and Food Help NYC can help now."
    )
    assert "{cite:S1}" in result.text
    assert "https://finder.nyc.gov/foodhelp" in result.text
    assert "https://foodhelp.nyc.gov" not in result.text
    assert result.status == "success"
    assert len(verifier.inputs) == 1
    assert verifier.inputs[0][0].claim.startswith("HRA can review")
    assert result.diagnostics["validation_rejections"] == [{
        "attempt": 1,
        "stage": "semantic_grounding",
        "items": [{"id": "claim-0", "label": "partial", "citation_ids": ["S1"]}],
        "citation_ids": ["S1"],
    }]


async def test_supported_multi_source_claims_remain_unchanged() -> None:
    answer = (
        "Food Help NYC can help now. {cite:S1}\n\n"
        "Contact HRA to ask about the case decision. {cite:S2}"
    )
    verifier = _OwnershipVerifier(partial=False)

    result, calls = await _run(answer, verifier)

    assert calls == 2
    assert result.text == answer
    assert result.status == "success"
    assert len(verifier.inputs[0]) == 2
    assert result.diagnostics["validation_rejections"] == []
    assert result.usage["model_time_ms"] >= 12.0


async def test_partial_grounded_block_keeps_text_and_source_lead() -> None:
    verifier = _OwnershipVerifier(partial=True)

    result, _calls = await _run(
        "Call 311 for Food Help NYC.",
        verifier,
        grounded=True,
    )

    assert "Call 311 for Food Help NYC." in result.text
    assert "{cite:S1}" in result.text
    assert "https://finder.nyc.gov/foodhelp" in result.text


@pytest.mark.parametrize(
    "unsupported_claim",
    [
        "Unsupported sentence {cite:S1}.",
        "Unsupported sentence. {cite:S1}",
    ],
)
async def test_rejected_plain_claim_keeps_supported_sibling_citation(
    unsupported_claim: str,
) -> None:
    class MixedVerifier(_OwnershipVerifier):
        async def arun_many(self, inputs):
            self.inputs.append(inputs)
            return NLIBatchRun([
                NLIVerdict(
                    supported=index > 0,
                    score=float(index > 0),
                    backend="fake",
                    label="supported" if index > 0 else "unsupported",
                )
                for index, _item in enumerate(inputs)
            ])

    result, _calls = await _run(
        f"{unsupported_claim} Supported sentence {{cite:S1}}.",
        MixedVerifier(partial=False),
    )

    assert "Unsupported sentence." in result.text
    assert "Supported sentence {cite:S1}." in result.text
    assert result.text.count("{cite:S1}") == 1
