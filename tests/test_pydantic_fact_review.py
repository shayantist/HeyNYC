from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter, build_runtime
from heynyc.core.pydantic_runtime.tools import (
    ResidentFactReviewCapability,
    _fact_review_prompt,
    _resident_review_schema,
    resident_fact_confirmation_tool,
    runtime_tool,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext, ToolInput


def test_fact_review_prompt_redacts_identifiers_before_second_model() -> None:
    prompt = _fact_review_prompt(
        ("I am 35.", "My SSN is 123-45-6789 and I live at 12 Main St.")
    )

    assert "123-45-6789" not in prompt
    assert "12 Main St" not in prompt
    assert prompt.count("[redacted]") == 2


async def test_fact_review_preserves_false_and_omits_unknown_before_execution() -> None:
    class Person(ToolInput):
        age: float
        pregnant: bool | None = None
        veteran: bool | None = None

    class Input(ToolInput):
        persons: list[Person]

    executed: list[dict] = []

    async def handler(args: dict, ctx: ToolContext) -> str:
        executed.append(args)
        return "screened"

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        input_type=Input,
        handler=handler,
        resident_fact_scope=("/persons",),
    )
    confirmation = resident_fact_confirmation_tool(source)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await confirmation.invoke(
        {"persons": [{"age": 35, "pregnant": False}]},
        ctx,
    )

    assert result == "screened"
    assert confirmation.input_type is source.input_type
    assert executed[0].model_dump(exclude_none=True) == {
        "persons": [{"age": 35.0, "pregnant": False}]
    }
    assert ctx.resident_facts["/persons/0/pregnant"].value is False
    assert "/persons/0/veteran" not in ctx.resident_facts


def test_fact_review_schema_requires_explicit_null_for_unknown_scoped_facts() -> None:
    class Profile(ToolInput):
        age: float
        pregnant: bool | None = None
        status: Literal["working", "retired"] | None = None

    class Input(ToolInput):
        profile: Profile
        goal: str | None = None

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        input_type=Input,
        handler=lambda args, ctx: None,
        resident_fact_scope=("/profile",),
    )

    schema = _resident_review_schema(source)
    profile = schema["properties"]["profile"]

    assert schema["required"] == ["profile"]
    assert "goal" not in schema["properties"]
    assert profile["required"] == ["age", "pregnant", "status"]
    assert profile["properties"]["age"]["type"] == "number"
    assert profile["properties"]["pregnant"]["anyOf"] == [
        {"type": "boolean"},
        {"type": "null"},
    ]
    assert profile["properties"]["status"]["anyOf"] == [
        {"type": "string", "enum": ["working", "retired"]},
        {"type": "null"},
    ]
    source_profile = source._input_schema()["$defs"]["Profile"]
    assert source_profile["required"] == ["age"]


async def test_fact_review_normalizes_confirmation_before_resident_approval() -> None:
    class Profile(ToolInput):
        age: float
        pregnant: bool | None = None
        veteran: bool | None = None

    class Input(ToolInput):
        profile: Profile
        goal: str | None = None

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        input_type=Input,
        handler=lambda args, ctx: None,
        resident_fact_scope=("/profile",),
    )
    confirmation = resident_fact_confirmation_tool(source)

    async def answer_model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([
            ToolCallPart(
                confirmation.name,
                {"profile": {"age": 35}, "goal": "food tonight"},
                "review",
            )
        ])

    review_settings = []
    review_thinking = []

    async def review_model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        review_settings.append(info.model_settings)
        review_thinking.append(info.model_request_parameters.thinking)
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "profile": {
                        "age": 35,
                        "pregnant": False,
                        "veteran": None,
                    },
                },
            )
        ])

    agent = Agent(
        FunctionModel(answer_model),
        deps_type=ToolContext,
        tools=[runtime_tool(confirmation)],
        capabilities=[
            ResidentFactReviewCapability(
                FunctionModel(
                    review_model,
                    profile={"supports_thinking": True},
                ),
                model_name="review-model",
                governed={confirmation.name: source},
            )
        ],
        output_type=[str, DeferredToolRequests],
    )
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        user_turns=("I am 35 and not pregnant.",),
    )

    result = await agent.run("Screen me", deps=ctx)

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals[0].args_as_dict() == {
        "profile": {"age": 35, "pregnant": False},
        "goal": "food tonight",
    }
    assert review_settings == [{"timeout": 30}]
    assert review_thinking == ["low"]


async def test_runtime_accounts_for_fact_review_model_call() -> None:
    class Profile(ToolInput):
        age: float
        pregnant: bool | None = None

    class Input(ToolInput):
        profile: Profile

    async def handler(args: dict, ctx: ToolContext) -> str:
        return "screened"

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        input_type=Input,
        handler=handler,
        resident_fact_scope=("/profile",),
    )

    async def answer_model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([
            ToolCallPart(
                "confirm_screen_facts",
                {"profile": {"age": 35, "pregnant": None}},
                "review",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer_model),
        registry=Registry([]),
        tools={source.name: source},
        guard_grounding=False,
        fact_review_model=TestModel(
            custom_output_args={
                "profile": {"age": 35, "pregnant": False},
            }
        ),
        fact_review_model_name="openai/gpt-5.4-nano",
    )

    conversation = runtime.conversation()
    result = await conversation.send(
        "I am 35 and not pregnant. Screen me."
    )

    assert result.status == "approval_required"
    assert conversation.pending_approvals["review"]["args"] == {
        "profile": {"age": 35, "pregnant": False}
    }
    assert result.usage["n_model_calls"] == 2
    assert result.usage["fact_review_model"] == "openai/gpt-5.4-nano"


def test_public_runtime_governs_resident_fact_tools_by_default() -> None:
    class Input(ToolInput):
        profile: dict

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        input_type=Input,
        handler=lambda args, ctx: None,
        resident_fact_scope=("/profile",),
    )

    runtime = build_runtime(
        Registry([]),
        model=TestModel(),
        tools={source.name: source},
    )

    assert any(
        isinstance(capability, ResidentFactReviewCapability)
        for capability in runtime._agent.root_capability.capabilities
    )
