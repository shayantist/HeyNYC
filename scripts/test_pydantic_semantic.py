from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest import MonkeyPatch

from heynyc.core.nli import NLIBatchRun, NLIVerdict
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from scripts import pydantic_ai_ab
from scripts.pydantic_ai_ab import _parser, build_factories
from scripts.pydantic_ai_parity import PydanticRuntimeAdapter


class _Verifier:
    def __init__(self) -> None:
        self.inputs = []

    async def arun_many(self, inputs):
        self.inputs.append(inputs)
        supported = len(self.inputs) > 1
        return NLIBatchRun(
            verdicts=[
                NLIVerdict(
                    supported=supported,
                    score=float(supported),
                    backend="fake",
                    reason="" if supported else "claim is broader than the evidence",
                    label="supported" if supported else "partial",
                )
                for _ in inputs
            ],
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
            latency_ms=10,
        )


class _FieldVerifier:
    def __init__(self) -> None:
        self.inputs = []

    async def arun_many(self, inputs):
        self.inputs.append(inputs)
        verdicts = [
            NLIVerdict(
                supported="benefits end tomorrow" not in item.claim.lower(),
                score=1.0 if "benefits end tomorrow" not in item.claim.lower() else 0.0,
                backend="fake",
                label=(
                    "supported"
                    if "benefits end tomorrow" not in item.claim.lower()
                    else "unsupported"
                ),
            )
            for item in inputs
        ]
        return NLIBatchRun(verdicts=verdicts)


class _FailingVerifier:
    async def arun_many(self, inputs):
        return NLIBatchRun(
            verdicts=[
                NLIVerdict(False, 0.0, "fake", "provider unavailable", "unsupported")
                for _ in inputs
            ],
            error="RuntimeError",
        )


async def test_pydantic_semantic_validator_retries_and_accounts_for_verifier() -> None:
    async def handler(args: dict, ctx: ToolContext) -> str:
        cid = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="NYC businesses must accept cash unless a stated exception applies.",
        )
        return f"NYC businesses must accept cash unless an exception applies. {{cite:{cid}}}"

    source = Tool(
        name="official_guidance",
        description="Get current official guidance",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    model_calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("official_guidance", {}, "guidance-1")])
        claim = (
            "Every business must always accept cash."
            if model_calls == 2
            else "NYC businesses must accept cash unless a stated exception applies."
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {"text": claim, "citation_ids": ["S1"]},
                    ],
                },
                f"final-{model_calls}",
            )
        ])

    verifier = _Verifier()
    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={source.name: source},
        structured_grounding=True,
        semantic_verifier=verifier,
    )

    result = await runtime.run("Do businesses have to accept cash?")

    assert model_calls == 3
    assert len(verifier.inputs) == 2
    assert verifier.inputs[0][0].claim == "Every business must always accept cash."
    assert "unless a stated exception applies" in verifier.inputs[0][0].source
    assert result.text == (
        "NYC businesses must accept cash unless a stated exception applies. {cite:S1}"
    )
    assert result.usage["semantic_verifier_requests"] == 2
    assert result.usage["semantic_verifier_input_tokens"] == 200
    assert result.usage["semantic_verifier_output_tokens"] == 40
    assert result.usage["semantic_verifier_cost_usd"] == 0.002
    assert result.usage["semantic_verifier_time_ms"] == 20
    assert result.usage["semantic_verifier_labels"] == {
        "partial": 1,
        "supported": 1,
    }
    assert result.usage["input_tokens"] == result.usage["answer_input_tokens"] + 200
    assert result.usage["output_tokens"] == result.usage["answer_output_tokens"] + 40
    assert result.usage["requests"] == result.usage["n_answer_model_calls"] + 2


async def test_pydantic_semantic_validator_checks_conversational_fields() -> None:
    model_calls = 0

    async def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        unsupported = model_calls == 1
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "acknowledgment": (
                        "Your benefits end tomorrow."
                        if unsupported
                        else "I'm sorry you're dealing with this."
                    ),
                    "follow_up_question": (
                        "Your benefits end tomorrow. What is your ZIP code?"
                        if unsupported
                        else "What is your ZIP code?"
                    ),
                },
                f"final-{model_calls}",
            )
        ])

    verifier = _FieldVerifier()
    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
        semantic_verifier=verifier,
    )

    result = await runtime.run("I need help with my benefits.")

    assert model_calls == 2
    assert [item.id for item in verifier.inputs[0]] == [
        "acknowledgment",
        "follow-up-question",
    ]
    assert all(
        item.source == "Resident message:\nI need help with my benefits."
        for item in verifier.inputs[0]
    )
    assert result.text == (
        "I'm sorry you're dealing with this.\n\nWhat is your ZIP code?"
    )


async def test_pydantic_semantic_validator_fails_safely_when_provider_is_down() -> None:
    model_calls = 0

    async def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([
                ToolCallPart("official_guidance", {}, "guidance-1")
            ])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {"text": "Your benefits end tomorrow.", "citation_ids": ["S1"]},
                    ],
                },
                "final-1",
            )
        ])

    async def source(_args: dict, ctx: ToolContext) -> str:
        cid = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits are available.",
        )
        return f"Benefits are available. {{cite:{cid}}}"

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "official_guidance": Tool(
                name="official_guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        semantic_verifier=_FailingVerifier(),
    )

    result = await runtime.run("Will my benefits end?")

    assert result.text == (
        "I'm sorry, I couldn't verify the sources needed to answer safely right now. "
        "Please try again."
    )
    assert result.usage["semantic_verifier_error"] == "RuntimeError"


def test_ab_cli_passes_semantic_verifier_only_to_candidate(
    monkeypatch: MonkeyPatch,
) -> None:
    args = _parser().parse_args([
        "--arm",
        "pydantic_ai",
        "--case",
        "example",
        "--structured-grounding",
        "--semantic-verifier-model",
        "openai/gpt-5.4-nano",
    ])
    assert args.semantic_verifier_model == "openai/gpt-5.4-nano"

    verifier = _Verifier()
    monkeypatch.setattr(pydantic_ai_ab, "_comparison_model", lambda _model: TestModel())
    factories = build_factories(
        Registry([]),
        None,
        "openai/gpt-5.4-mini",
        structured_grounding=True,
        semantic_verifier=verifier,
    )
    candidate = factories["pydantic_ai"]()
    assert candidate.runtime._semantic_verifier is verifier
