from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import PydanticRunFailure
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_default_event_shortlist_retries_for_a_useful_next_step() -> None:
    calls = 0

    async def events(_args: dict, _ctx: ToolContext) -> str:
        return "One event. This is a shortlist, not every matching event."

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("find_nyc_events", {}, "events-1")])
        answer = (
            "Here is one option from a short list."
            if calls == 2
            else "Here is one option from a short list. Want more music choices?"
        )
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

    assert result.text.endswith("Want more music choices?")
    assert result.diagnostics["validation_rejections"] == [
        {"attempt": 1, "stage": "shortlist_next_step"}
    ]


async def test_exhausted_shortlist_followup_is_labeled_incomplete_not_unverified() -> None:
    calls = 0

    async def events(_args: dict, _ctx: ToolContext) -> str:
        return "One event. This is a shortlist, not every matching event."

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("find_nyc_events", {}, "events-1")])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {"answer": "Here is one grounded option."},
                f"answer-{calls}",
            )
        ])

    with pytest.raises(PydanticRunFailure) as raised:
        await PydanticRuntimeAdapter(
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

    text = raised.value.partial_result.text
    assert "couldn't complete every requested part" in text
    assert "Here is one grounded option." in text
    assert "couldn't verify every detail" not in text


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
