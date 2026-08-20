from __future__ import annotations

from types import SimpleNamespace

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_default_event_shortlist_does_not_retry_for_optional_followup_copy() -> None:
    calls = 0

    async def events(_args: dict, _ctx: ToolContext) -> str:
        return "One event. This is a shortlist, not every matching event."

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("find_nyc_events", {}, "events-1")])
        answer = "Here is one option from a short list."
        return ModelResponse([ToolCallPart("final_answer", {"answer": answer}, f"answer-{calls}")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "find_nyc_events": Tool(
                name="find_nyc_events",
                description="Find NYC events",
                parameters={"type": "object", "properties": {}},
                handler=events,
            )
        },
        structured_grounding=True,
    ).run("What can I do today?")

    assert result.text == "Here is one option from a short list."
    assert calls == 2
    assert result.diagnostics["validation_rejections"] == []


async def test_explicit_event_count_does_not_require_a_follow_up_question() -> None:
    calls = 0

    async def events(_args: dict, _ctx: ToolContext) -> str:
        return "One requested event."

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("find_nyc_events", {}, "events-1")])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {"answer": "Here is the one event you requested."},
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "find_nyc_events": Tool(
                name="find_nyc_events",
                description="Find NYC events",
                parameters={"type": "object", "properties": {}},
                handler=events,
            )
        },
        structured_grounding=True,
    ).run("Give me one event today")

    assert result.text == "Here is the one event you requested."
    assert calls == 2


async def test_english_answer_retries_unexpected_unsourced_script() -> None:
    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        answer = (
            "Here are ten distinct events. เว็บพนัน"
            if calls == 1
            else "Here are ten distinct events."
        )
        return ModelResponse([TextPart(answer)])

    async def screen(_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="none",
            language="en",
            model="test/safety",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
    ).run("Give me ten events today")

    assert result.text == "Here are ten distinct events."
    assert result.diagnostics["validation_rejections"] == [
        {"attempt": 1, "stage": "unexpected_script", "scripts": ["THAI"]}
    ]


async def test_english_answer_allows_a_source_backed_non_latin_name() -> None:
    async def events(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://example.com/thai-festival",
            title="Thai Festival",
            kind="WEB",
            snippet="The listing names เทศกาลไทย.",
            provenance={"source_tier": "authoritative", "evidence_grade": "authoritative"},
        )
        return f"The listing names เทศกาลไทย. {{cite:{citation_id}}}"

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("events", {}, "events-1")])
        return ModelResponse([TextPart("The listing names เทศกาลไทย. {cite:S1}")])

    async def screen(_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="none",
            language="en",
            model="test/safety",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "events": Tool(
                name="events",
                description="Find events",
                parameters={"type": "object", "properties": {}},
                handler=events,
            )
        },
        crisis_screen=screen,
    ).run("What events are happening today?")

    assert "เทศกาลไทย" in result.text
    assert result.diagnostics["validation_rejections"] == []
