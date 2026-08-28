import asyncio

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.channels.format import render
from heynyc.core.pydantic_runtime import PydanticRunFailure, PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_stalled_final_request_continues_from_completed_tools() -> None:
    model_calls = 0
    tool_calls = 0

    async def lookup(_args: dict, ctx: ToolContext) -> str:
        nonlocal tool_calls
        tool_calls += 1
        citation_id = ctx.citations.register(
            "https://official.example/record",
            title="Official record",
            kind="DATA",
            snippet="The record is open.",
        )
        return f"The record is open. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        if model_calls == 2:
            await asyncio.sleep(1)
        assert not info.function_tools
        output = next(
            tool for tool in info.output_tools if tool.name == "final_answer"
        )
        return ModelResponse(
            [
                ToolCallPart(
                    output.name,
                    {"answer": "The official record is open. {cite:S1}"},
                    "answer-1",
                )
            ]
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up the official record",
                handler=lookup,
            )
        },
        structured_grounding=True,
        model_request_timeout_s=0.01,
    )

    result = await runtime.run("What does the record show?")

    assert result.status == "success"
    assert result.text.startswith("The official record is open.")
    assert "Verified source" not in result.text
    assert tool_calls == 1
    assert model_calls == 3
    assert result.usage["stalled_model_requests"] == 1
    assert result.usage["stalled_model_recoveries"] == 1


async def test_stalled_followup_preserves_an_existing_cited_draft_without_retry() -> None:
    model_calls = 0

    async def lookup_one(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://official.example/one",
            title="Official record",
            kind="DATA",
            snippet="Record one is open.",
        )
        return f"Record one is open. {{cite:{citation_id}}}"

    async def lookup_two(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://official.example/two",
            title="Official record",
            kind="DATA",
            snippet="Record two is open.",
        )
        return f"Record two is open. {{cite:{citation_id}}}"

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("lookup_one", {}, "lookup-1")])
        if len(returns) == 1:
            return ModelResponse([
                TextPart("Record one is open. {cite:S1}"),
                ToolCallPart("lookup_two", {}, "lookup-2"),
            ])
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup_one": Tool(
                name="lookup_one",
                description="Look up the first official record",
                handler=lookup_one,
            ),
            "lookup_two": Tool(
                name="lookup_two",
                description="Look up the second official record",
                handler=lookup_two,
            )
        },
        structured_grounding=True,
        model_request_timeout_s=0.01,
    )

    with pytest.raises(PydanticRunFailure) as caught:
        await runtime.run("What do the records show?")

    assert caught.value.partial_result.text.startswith("Record one is open. {cite:S1}")
    assert model_calls == 3
    assert caught.value.partial_result.usage["stalled_model_requests"] == 1


async def test_repeated_stall_preserves_evidence_without_internal_source_labels() -> (
    None
):
    async def lookup(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://official.example/record",
            title="Official record",
            kind="DATA",
            snippet="The record is open.",
        )
        excerpt_id = ctx.citations.register(
            "https://official.example/help",
            title="Tenant organizing help",
            kind="WEB",
            snippet="Tenant organizers can help residents form an association.",
            provenance={
                "evidence_grade": "authoritative_excerpt",
                "search": {"provider": "Tavily Search API", "score": 0.63},
            },
        )
        return f"The record is open. {{cite:{citation_id}}} {{cite:{excerpt_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up the official record",
                handler=lookup,
            )
        },
        structured_grounding=True,
        model_request_timeout_s=0.01,
    )

    with pytest.raises(PydanticRunFailure) as caught:
        await runtime.run("What does the record show?")

    text = "\n".join(render(caught.value.partial_result, "sms_twilio"))
    assert "The record is open." in text
    assert "Tenant organizing help" in text
    assert "https://official.example/record" in text
    assert "https://official.example/help" in text
    assert "Verified source" not in text
    assert "Unverified source" not in text
    assert "Official search excerpt" not in text
    assert "Provider relevance score" not in text


async def test_run_wall_timeout_does_not_start_answer_recovery() -> None:
    model_calls = 0

    async def lookup(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://official.example/record",
            title="Official record",
            kind="DATA",
            snippet="The record is open.",
        )
        return f"The record is open. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up the official record",
                handler=lookup,
            )
        },
        structured_grounding=True,
        run_timeout_s=0.05,
        model_request_timeout_s=5,
    )

    with pytest.raises(PydanticRunFailure) as caught:
        await runtime.run("What does the record show?")

    assert model_calls == 2
    assert "stalled_model_requests" not in caught.value.partial_result.usage
    assert "stalled_model_recoveries" not in caught.value.partial_result.usage
