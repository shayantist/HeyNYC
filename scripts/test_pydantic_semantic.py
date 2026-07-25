from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest import MonkeyPatch

from heynyc.core.nli import NLIBatchRun, NLIVerdict
from heynyc.core.pydantic_runtime import (
    PydanticRuntimeAdapter,
    _semantic_citation_evidence,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.cases import EvalCase
from heynyc.eval.runner import run_case
from scripts import pydantic_ai_ab
from scripts.pydantic_ai_ab import _parser, build_factories


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
                    reason="" if supported else "private source detail",
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


def test_semantic_evidence_never_falls_back_to_full_provenance() -> None:
    assert _semantic_citation_evidence({
        "provenance": {
            "snapshot": {"private_irrelevant_field": "must stay local"}
        }
    }) == ""


def test_semantic_evidence_is_bounded() -> None:
    evidence = _semantic_citation_evidence({
        "snippet": f"{'x' * 2_000} private-tail",
        "title": "Official guidance",
    })

    assert len(evidence) <= 1_200
    assert "private-tail" not in evidence


async def test_structured_answer_contract_requires_direct_useful_answer() -> None:
    description = ""
    system_prompts: list[str] = []
    output_schema: dict = {}

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal description, output_schema
        description = info.output_tools[0].description or ""
        output_schema = info.output_tools[0].parameters_json_schema
        system_prompts.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        )
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {"grounded_blocks": []},
                "final-1",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
        guard_grounding=False,
    )

    await runtime.run("Will my benefits stop?")

    normalized = description.lower()
    assert "answer the resident's actual question first" in normalized
    assert "first grounded block" in normalized
    assert "not in the acknowledgment" in normalized
    assert "individual outcome" in normalized
    assert "concrete official next step" in normalized
    assert "state that limitation in the acknowledgment" in normalized
    acknowledgment_description = (
        output_schema["properties"]["acknowledgment"]["description"].lower()
    )
    assert "explicit limitation on what can be determined" in acknowledgment_description
    assert "external factual or procedural claims" in acknowledgment_description
    follow_up_description = (
        output_schema["properties"]["follow_up_question"]["description"].lower()
    )
    assert "neutral clarification question" in follow_up_description
    assert "data-minimization reminder" in follow_up_description
    assert any(
        "put retrieved source ids only in citation_ids" in prompt.lower()
        for prompt in system_prompts
    )
    assert any(
        "do not write inline citation markers" in prompt.lower()
        for prompt in system_prompts
    )


async def test_structured_answer_normalizes_matching_embedded_citation_marker() -> None:
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits may change based on the resident's circumstances.",
        )
        return f"Benefits depend on the resident's circumstances. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {
                            "text": "Benefits may change. {cite:S1}",
                            "citation_ids": ["S1"],
                        },
                    ]
                },
                "final-1",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
    )

    result = await runtime.run("Will my benefits change?")

    assert model_calls == 2
    assert result.text == "Benefits may change. {cite:S1}"


async def test_structured_answer_retries_extra_declared_id_with_embedded_marker() -> None:
    async def source(_args: dict, ctx: ToolContext) -> str:
        first = ctx.citations.register(
            "https://www.nyc.gov/first",
            title="First source",
            kind="WEB",
            snippet="Benefits may change.",
        )
        second = ctx.citations.register(
            "https://www.nyc.gov/second",
            title="Second source",
            kind="WEB",
            snippet="Unrelated guidance.",
        )
        return f"Benefits may change. {{cite:{first}}} Other guidance. {{cite:{second}}}"

    model_calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        if model_calls == 3:
            feedback = str(
                [
                    part.content
                    for message in messages
                    for part in message.parts
                    if part.part_kind == "retry-prompt"
                ][-1]
            )
            assert "S2" not in feedback
        citation_ids = ["S1", "S2"] if model_calls == 2 else ["S1"]
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {
                            "text": "Benefits may change. {cite:S1}",
                            "citation_ids": citation_ids,
                        },
                    ]
                },
                f"final-{model_calls}",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
    )

    result = await runtime.run("Will my benefits change?")

    assert model_calls == 3
    assert result.text == "Benefits may change. {cite:S1}"


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
        if model_calls == 3:
            feedback = str(
                [
                    part.content
                    for message in messages
                    for part in message.parts
                    if part.part_kind == "retry-prompt"
                ][-1]
            )
            assert "private source detail" not in feedback
            assert "block-0" not in feedback
            assert "partial" not in feedback
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


async def test_pydantic_semantic_validator_uses_framing_and_bounded_citation_chunks() -> None:
    async def handler(args: dict, ctx: ToolContext) -> str:
        cid = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="DATA",
            snippet="Eligibility depends on household circumstances.",
            provenance={
                "snapshot": {
                    "private_irrelevant_field": "raw row content must not reach semantic NLI"
                }
            },
        )
        return f"Eligibility depends on household circumstances. {{cite:{cid}}}"

    source = Tool(
        name="official_guidance",
        description="Get current official guidance",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    verifier = _FieldVerifier()
    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("official_guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "acknowledgment": (
                        "I understand why you are worried. "
                        "I cannot tell from this message alone."
                    ),
                    "grounded_blocks": [
                        {
                            "text": "Eligibility depends on household circumstances.",
                            "citation_ids": ["S1"],
                        }
                    ],
                },
                "final-1",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={source.name: source},
        structured_grounding=True,
        semantic_verifier=verifier,
    )

    await runtime.run("Will my benefits stop?")

    inputs = verifier.inputs[0]
    acknowledgment = next(item for item in inputs if item.id == "acknowledgment")
    grounded = next(item for item in inputs if item.id == "block-0")
    assert acknowledgment.kind == "framing"
    assert acknowledgment.source == ""
    assert grounded.source == (
        "[S1] Eligibility depends on household circumstances. Official guidance"
    )
    assert "private_irrelevant_field" not in grounded.source


async def test_semantic_verifier_bounds_total_evidence_per_claim() -> None:
    async def handler(_args: dict, ctx: ToolContext) -> str:
        citation_ids = [
            ctx.citations.register(
                f"https://www.nyc.gov/example-{index}",
                title=f"Official guidance {index}",
                kind="WEB",
                snippet=f"{index} " + "x" * 800,
            )
            for index in range(4)
        ]
        return " ".join(f"{{cite:{citation_id}}}" for citation_id in citation_ids)

    verifier = _FieldVerifier()
    calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {
                            "text": "Eligibility depends on household circumstances.",
                            "citation_ids": ["S1", "S2", "S3", "S4"],
                        }
                    ],
                },
                "final-1",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        },
        structured_grounding=True,
        semantic_verifier=verifier,
    )

    await runtime.run("Will my benefits change?")

    grounded = next(item for item in verifier.inputs[0] if item.kind == "claim")
    assert len(grounded.source) <= 1_200
    assert "[S1]" in grounded.source


async def test_failed_eval_keeps_semantic_diagnostics_out_of_usage() -> None:
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits may change.",
        )
        return f"Benefits may change. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [
                        {
                            "text": "Benefits may change.",
                            "citation_ids": ["S1"],
                        }
                    ],
                },
                f"final-{model_calls}",
            )
        ])

    class RejectingVerifier:
        async def arun_many(self, inputs):
            return NLIBatchRun(
                verdicts=[
                    NLIVerdict(
                        False,
                        0.0,
                        "fake",
                        "private diagnostic reason",
                        "unsupported",
                    )
                    for _ in inputs
                ]
            )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        semantic_verifier=RejectingVerifier(),
    )

    result = await run_case(
        runtime,
        EvalCase(
            id="semantic-failure",
            module="parity",
            query="Will my benefits change?",
        ),
    )

    assert result.error == "Exceeded maximum output retries (2)"
    item = result.diagnostics["semantic_verifier_runs"][0]["items"][0]
    assert item == {
        "position": 0,
        "kind": "claim",
        "label": "unsupported",
    }
    assert "Benefits may change." not in str(result.diagnostics)
    assert "private diagnostic reason" not in str(result.usage)
    assert "private diagnostic reason" not in str(result.diagnostics)


async def test_pydantic_semantic_validator_checks_conversational_fields() -> None:
    model_calls = 0

    async def source(_args: dict, ctx: ToolContext) -> str:
        cid = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits depend on household circumstances.",
        )
        return f"Benefits depend on household circumstances. {{cite:{cid}}}"

    async def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        unsupported = model_calls == 2
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
                    "grounded_blocks": [
                        {
                            "text": "Benefits depend on household circumstances.",
                            "citation_ids": ["S1"],
                        }
                    ],
                },
                f"final-{model_calls}",
            )
        ])

    verifier = _FieldVerifier()
    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        semantic_verifier=verifier,
    )

    result = await runtime.run("I need help with my benefits.")

    assert model_calls == 3
    assert [item.id for item in verifier.inputs[0]] == [
        "acknowledgment",
        "block-0",
        "follow-up-question",
    ]
    assert verifier.inputs[0][0].source == ""
    assert verifier.inputs[0][2].source == ""
    assert result.text == (
        "I'm sorry you're dealing with this.\n\n"
        "Benefits depend on household circumstances. {cite:S1}\n\n"
        "What is your ZIP code?"
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
