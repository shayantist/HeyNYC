from types import SimpleNamespace

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.faults import VerificationFallbackProbeModel, verified_fallback_probe


@pytest.mark.asyncio
async def test_verification_fallback_probe_exhausts_real_validation_path() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official source supports a different claim.",
        )
        return f"Official evidence. {{cite:{citation_id}}}"

    delegated_calls = 0

    async def delegate(_messages, _info: AgentInfo) -> ModelResponse:
        nonlocal delegated_calls
        delegated_calls += 1
        return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])

    async def language_screen(_user_turns):
        return SimpleNamespace(
            risk="none",
            language="es",
            model="test/safety",
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            requests=0,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    runtime = PydanticRuntimeAdapter(
        VerificationFallbackProbeModel(FunctionModel(delegate)),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve official evidence",
                parameters={"type": "object", "properties": {}},
                handler=retrieve,
            )
        },
        structured_grounding=True,
        crisis_screen=language_screen,
    )

    result = await runtime.run("¿Puedes verificar este número?")

    assert delegated_calls == 1
    assert result.status == "success"
    assert result.text.startswith("No pude verificarlo con las fuentes confiables")
    assert result.diagnostics["safety_language"] == "es"
    assert [item["stage"] for item in result.diagnostics["validation_rejections"]] == [
        "deterministic_grounding",
        "deterministic_grounding",
        "deterministic_grounding",
    ]
    assert verified_fallback_probe(result, result.text)
    result.text += "\n\nPáginas oficiales a las que sí pude acceder antes del problema:"
    assert not verified_fallback_probe(
        result,
        "No pude verificarlo con las fuentes confiables",
    )
