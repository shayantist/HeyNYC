import time
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from heynyc.channels.format import render
from heynyc.core.citations import CitationRegistry, data_provenance
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext

LIMITATION = (
    "These are regular hours. Confirm holiday or temporary schedule exceptions before traveling."
)
SPANISH_LIMITATION = (
    "Estas son las horas habituales. Confirme las excepciones por días festivos o cambios "
    "temporales en el horario antes de viajar."
)


def _project(*, limitation: str = "", language: str | None = None):
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
        f"Queens SNAP Center has regular weekday hours {{cite:{citation_id}}}",
        citations,
        time.perf_counter(),
        model_time_ms=0,
        language=language,
    )
    if language is not None:
        result.diagnostics["safety_language"] = language
    return result


def test_f176_channel_renders_the_complete_english_source_limit_once() -> None:
    result = _project(limitation=LIMITATION)

    assert LIMITATION not in result.text
    assert render(result, "sms_twilio")[0].count(LIMITATION) == 1


def test_f176_channel_uses_the_complete_spanish_catalog_message() -> None:
    result = _project(limitation=LIMITATION, language="es")

    assert SPANISH_LIMITATION not in result.text
    rendered = render(result, "sms_twilio")[0]
    assert SPANISH_LIMITATION in rendered
    assert "Fuentes:" in rendered
    assert "Nota de la fuente -" in rendered
    assert "Sources:" not in rendered
    assert "Source note -" not in rendered


def test_f176_projection_does_not_add_a_limit_without_one() -> None:
    result = _project()

    assert LIMITATION not in result.text
    assert LIMITATION not in render(result, "sms_twilio")[0]


def test_f176_projection_does_not_leak_english_without_a_locale_message() -> None:
    result = _project(limitation=LIMITATION, language="ht")

    assert LIMITATION not in result.text
    assert LIMITATION not in render(result, "sms_twilio")[0]


@pytest.mark.asyncio
async def test_f176_missing_localized_limit_is_rendered_without_a_model_retry() -> None:
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
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {
                    "answer": (
                        "Queens SNAP Center tiene un horario habitual {cite:S1}"
                    ),
                },
                "answer-1",
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
    assert SPANISH_LIMITATION not in result.text
    assert render(result, "sms_twilio")[0].count(SPANISH_LIMITATION) == 1
