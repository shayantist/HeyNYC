import time
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from heynyc.core.citations import CitationRegistry, data_provenance
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import GroundedAnswer, GroundedBlock
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext

LIMITATION = (
    "These are regular hours. Confirm holiday or temporary schedule exceptions before traveling."
)
SPANISH_LIMITATION = (
    "Estas son las horas habituales. Confirme las excepciones por días festivos o cambios "
    "temporales en el horario antes de viajar."
)


def _project(*, limitation: str = "", language: str | None = None) -> str:
    citations = CitationRegistry()
    citation_id = citations.register(
        "https://data.cityofnewyork.us/resource/tc6u-8rnp/S1.json",
        title="HRA SNAP Centers",
        snippet="Queens SNAP Center regular hours",
        kind="DATA",
        provenance=data_provenance(
            {"facility_name": "Queens SNAP Center"},
            record_id="S1",
            field_pointer="/facility_name",
            derivation={"limitations": limitation} if limitation else {},
        ),
    )
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda: "unused"),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
    )
    result = runtime._project_result(
        [],
        RunUsage(),
        GroundedAnswer(
            grounded_blocks=[
                GroundedBlock(
                    text="Queens SNAP Center has regular weekday hours",
                    citation_ids=[citation_id],
                )
            ]
        ),
        citations,
        time.perf_counter(),
        model_time_ms=0,
        language=language,
    )
    return result.text


def test_f176_projection_restores_the_complete_english_source_limit_once() -> None:
    text = _project(limitation=LIMITATION)

    assert text.count(LIMITATION) == 1


def test_f176_projection_uses_the_complete_spanish_catalog_message() -> None:
    assert SPANISH_LIMITATION in _project(limitation=LIMITATION, language="es")


def test_f176_projection_does_not_add_a_limit_without_one() -> None:
    assert LIMITATION not in _project()


def test_f176_projection_does_not_leak_english_without_a_locale_message() -> None:
    assert LIMITATION not in _project(limitation=LIMITATION, language="ht")


@pytest.mark.asyncio
async def test_f176_missing_localized_limit_is_added_without_a_model_retry() -> None:
    async def nearest(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://data.cityofnewyork.us/resource/tc6u-8rnp/S1.json",
            title="HRA SNAP Centers",
            snippet="Queens SNAP Center regular hours",
            kind="DATA",
            provenance=data_provenance(
                {"facility_name": "Queens SNAP Center"},
                record_id="S1",
                field_pointer="/facility_name",
                derivation={"limitations": LIMITATION},
            ),
        )
        return f"Queens SNAP Center regular hours {{cite:{citation_id}}}"

    calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("nearest", {}, "nearest-1")])
        text = "Queens SNAP Center tiene un horario habitual"
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {"grounded_blocks": [{"text": text, "citation_ids": ["S1"]}]},
                f"answer-{calls}",
            )
        ])

    async def crisis_screen(_turns):
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

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "nearest": Tool(
                name="nearest",
                description="Find a SNAP center",
                parameters={"type": "object", "properties": {}},
                handler=nearest,
            )
        },
        structured_grounding=True,
        crisis_screen=crisis_screen,
    ).run("¿Dónde puedo solicitar SNAP en persona cerca de Jackson Heights?")

    assert calls == 2
    assert result.diagnostics["validation_rejections"] == []
    assert result.text.count(SPANISH_LIMITATION) == 1
