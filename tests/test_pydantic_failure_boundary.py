import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import _resident_history
from heynyc.core.pydantic_runtime.runtime import PydanticRunFailure
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_structured_runtime_preserves_a_nonfactual_plain_text_decline() -> None:
    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        outcome = next(
            tool for tool in info.output_tools if tool.name == "nonfactual_outcome"
        )
        assert set(outcome.parameters_json_schema["properties"]) == {"kind"}
        return ModelResponse([
            ToolCallPart(
                outcome.name,
                {"kind": "unknowable"},
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).run("Who will win the next World Cup final?")

    assert result.status == "success"
    assert result.text == (
        "I can't know that yet. I can help with the practical NYC part instead."
    )
    assert result.tool_calls_made == []
    assert result.usage["requests"] == 1


async def test_nonfactual_outcome_remains_in_conversation_history() -> None:
    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            history = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, TextPart)
            ]
            assert history == [
                "I can't know that yet. I can help with the practical NYC part instead."
            ]
        return ModelResponse([
            ToolCallPart(
                next(
                    tool.name
                    for tool in info.output_tools
                    if tool.name == "nonfactual_outcome"
                ),
                {"kind": "unknowable"},
                f"answer-{calls}",
            )
        ])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).conversation()

    await conversation.send("Who will win the next World Cup final?")
    await conversation.send("Where could I watch it in NYC when it happens?")

    assert calls == 2


async def test_mechanical_guard_does_not_classify_uncited_prose() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official office is open on Mondays.",
        )
        return f"The office is open on Mondays. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([TextPart("The office is open on Mondays.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
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
    ).run("When is the office open?")

    assert result.text == "The office is open on Mondays."
    assert result.diagnostics["validation_rejections"] == []


async def test_authoritative_evidence_supports_native_cited_prose() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official office is open on Mondays.",
        )
        return f"The office is open on Mondays. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            TextPart("The official office is open on Mondays. {cite:S1}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
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
    ).run("When is the office open?")

    assert result.status == "success"
    assert result.text.startswith("The official office is open on Mondays.")


async def test_mechanical_guard_does_not_classify_prose_before_retrieval() -> None:
    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([TextPart("The office is open on Mondays.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).run("When is the office open?")

    assert result.text == "The office is open on Mondays."


@pytest.mark.parametrize(
    "follow_up",
    [
        "The office is open on Mondays.",
        "The office is open on Mondays; which time works for you?",
        "The office is open on Mondays?",
    ],
)
async def test_citation_free_clarification_cannot_include_factual_prose(
    follow_up: str,
) -> None:
    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([
            ToolCallPart(
                "grounded_answer",
                {
                    "grounded_blocks": [],
                    "follow_up_question": follow_up,
                },
                "answer-1",
            )
        ])

    with pytest.raises(PydanticRunFailure) as raised:
        await PydanticRuntimeAdapter(
            FunctionModel(model),
            registry=Registry([]),
            tools={},
            structured_grounding=True,
        ).run("When is the office open?")

    assert raised.value.partial_result.status == "error"
    assert follow_up not in raised.value.partial_result.text


async def test_accepted_structured_output_discards_sibling_plain_text() -> None:
    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert all(
                part.content != "The office is open on Mondays."
                for message in messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
        return ModelResponse([
            TextPart("The office is open on Mondays."),
            ToolCallPart(
                next(
                    tool.name
                    for tool in info.output_tools
                    if tool.name == "nonfactual_outcome"
                ),
                {"kind": "unknowable"},
                f"answer-{calls}",
            ),
        ])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).conversation()

    first = await conversation.send("Who will win the next World Cup final?")
    second = await conversation.send("What about the one after that?")

    assert first.text == (
        "I can't know that yet. I can help with the practical NYC part instead."
    )
    assert second.text == first.text


def test_history_projection_discards_sibling_plain_text() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("Who will win?")]),
        ModelResponse(parts=[
            TextPart("The office is open on Mondays."),
            ToolCallPart(
                "nonfactual_outcome",
                {"kind": "unknowable"},
                "answer-1",
            ),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(
                "nonfactual_outcome",
                "Final result processed.",
                "answer-1",
            )
        ]),
    ]

    assert _resident_history(messages) == [
        {"role": "user", "content": "Who will win?"},
        {
            "role": "assistant",
            "content": (
                "I can't know that yet. "
                "I can help with the practical NYC part instead."
            ),
        },
    ]


async def test_mechanical_boundary_does_not_parse_phone_semantics() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official source supports a different claim.",
        )
        return f"Official evidence. {{cite:{citation_id}}}"

    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            TextPart("Call the unsupported number 212-555-1212. {cite:S1}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
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
    ).run("Can you verify the number?")

    assert result.status == "success"
    assert result.text == "Call the unsupported number 212-555-1212. {cite:S1}"
    assert result.diagnostics["validation_rejections"] == []
