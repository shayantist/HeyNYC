from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.manifest import ServiceModule, SituationHint
from heynyc.core.nli import NLIBatchRun, NLIVerdict
from heynyc.core.pydantic_runtime.runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext

_HIGH_STAKES_REGISTRY = Registry([
    ServiceModule(
        name="benefits",
        situations=[SituationHint(
            name="benefits_guidance",
            definition="Guidance that can affect a resident's benefits.",
            high_stakes=True,
        )],
    )
])


async def _high_stakes_scope(_turns: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        model="test",
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        requests=0,
        cost_usd=0.0,
        latency_ms=0.0,
        modules=("benefits",),
        situations=("benefits_guidance",),
        event_turn=None,
    )


async def test_runtime_exposes_remaining_context_to_retrieval_tools() -> None:
    observed: list[int | None] = []

    async def probe(_args: dict, ctx: ToolContext) -> str:
        observed.append(ctx.evidence_token_budget)
        return "retrieved evidence"

    calls = 0

    async def model(_messages, _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("probe", {}, "probe-1")])
        return ModelResponse([TextPart("Done")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "probe": Tool(
                name="probe",
                description="Inspect the retrieval budget",
                parameters={"type": "object", "properties": {}},
                handler=probe,
            )
        },
        context_budget=100,
        measure_context=lambda history, _continuity: (
            40 if any(message.get("tool_calls") for message in history) else 30
        ),
        compact_context=lambda _history, _continuity: None,
    )

    await runtime.conversation().send("Find it")

    assert observed == [60]


async def test_claim_support_checker_receives_evidence_after_character_4000() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://nyc.gov/schedule",
            title="Official schedule",
            snippet=("Background material. " * 250) + "The final event ends at 5 PM.",
            provenance={"source_tier": "authoritative"},
        )
        return f"Official schedule. {{cite:{citation_id}}}"

    calls = 0

    async def model(_messages, _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            ToolCallPart(
                "grounded_answer",
                {
                    "grounded_blocks": [{
                        "text": "The final event ends at 5 PM.",
                        "citation_ids": ["S1"],
                    }]
                },
                "answer-1",
            )
        ])

    class Verifier:
        async def arun_many(self, inputs):
            assert "The final event ends at 5 PM." in inputs[0].source
            return NLIBatchRun(verdicts=[
                NLIVerdict(True, 1.0, "fake", label="supported")
            ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=_HIGH_STAKES_REGISTRY,
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve the schedule",
                parameters={"type": "object", "properties": {}},
                handler=retrieve,
            )
        },
        structured_grounding=True,
        claim_support_checker=Verifier(),
        scope_screen=_high_stakes_scope,
    ).run("When does it end?")

    assert result.text == "The final event ends at 5 PM. {cite:S1}"
