from __future__ import annotations

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter, build_runtime
from heynyc.core.pydantic_runtime.tools import (
    ResidentFactReviewCapability,
    _fact_review_prompt,
    _resident_collection_schema,
    _resident_review_schema,
    adapt_tool,
    resident_fact_confirmation_tool,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def test_fact_review_prompt_redacts_identifiers_before_second_model() -> None:
    prompt = _fact_review_prompt(
        ("I am 35.", "My SSN is 123-45-6789 and I live at 12 Main St.")
    )

    assert "123-45-6789" not in prompt
    assert "12 Main St" not in prompt
    assert prompt.count("[redacted]") == 2


async def test_fact_review_preserves_false_and_omits_unknown_before_execution() -> None:
    executed: list[dict] = []

    async def handler(args: dict, ctx: ToolContext) -> str:
        executed.append(args)
        return "screened"

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {
                "persons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "number"},
                            "pregnant": {"type": "boolean"},
                            "veteran": {"type": "boolean"},
                        },
                        "required": ["age"],
                    },
                }
            },
            "required": ["persons"],
        },
        handler=handler,
        resident_fact_scope=("/persons",),
    )
    confirmation = resident_fact_confirmation_tool(source)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await confirmation.handler(
        {"persons": [{"age": 35, "pregnant": False}]},
        ctx,
    )

    assert result == "screened"
    assert confirmation.parameters is source.parameters
    assert executed == [{"persons": [{"age": 35, "pregnant": False}]}]
    assert ctx.resident_facts["/persons/0/pregnant"].value is False
    assert "/persons/0/veteran" not in ctx.resident_facts


def test_fact_review_schema_requires_explicit_null_for_unknown_scoped_facts() -> None:
    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "number"},
                        "pregnant": {"type": "boolean"},
                        "status": {"type": "string", "enum": ["working", "retired"]},
                    },
                    "required": ["age"],
                },
                "goal": {"type": "string"},
            },
            "required": ["profile"],
        },
        handler=lambda args, ctx: None,
        resident_fact_scope=("/profile",),
    )

    schema = _resident_review_schema(source)
    profile = schema["properties"]["profile"]

    assert schema["required"] == ["profile"]
    assert "goal" not in schema["properties"]
    assert profile["required"] == ["age", "pregnant", "status"]
    assert profile["properties"]["age"] == {"type": "number"}
    assert profile["properties"]["pregnant"] == {
        "anyOf": [{"type": "boolean"}, {"type": "null"}]
    }
    assert profile["properties"]["status"] == {
        "anyOf": [
            {"type": "string", "enum": ["working", "retired"]},
            {"type": "null"},
        ]
    }
    assert source.parameters["properties"]["profile"]["required"] == ["age"]


def test_collection_schema_hides_optional_scoped_fields_from_answer_model() -> None:
    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {
                "household": {
                    "type": "object",
                    "properties": {"cash": {"type": "string"}},
                },
                "persons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "number"},
                            "pregnant": {"type": "boolean"},
                        },
                        "required": ["age"],
                    },
                },
                "goal": {"type": "string"},
            },
            "required": ["persons"],
        },
        handler=lambda args, ctx: None,
        resident_fact_scope=("/household", "/persons"),
    )

    schema = _resident_collection_schema(source)

    assert set(schema["properties"]) == {"persons", "goal"}
    assert set(schema["properties"]["persons"]["items"]["properties"]) == {"age"}


async def test_fact_review_normalizes_confirmation_before_resident_approval() -> None:
    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "number"},
                        "pregnant": {"type": "boolean"},
                        "veteran": {"type": "boolean"},
                    },
                    "required": ["age"],
                },
                "goal": {"type": "string"},
            },
            "required": ["profile"],
        },
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

    agent = Agent(
        FunctionModel(answer_model),
        deps_type=ToolContext,
        tools=[adapt_tool(confirmation)],
        capabilities=[
            ResidentFactReviewCapability(
                TestModel(
                    custom_output_args={
                        "profile": {
                            "age": 35,
                            "pregnant": False,
                            "veteran": None,
                        },
                    }
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


async def test_runtime_accounts_for_fact_review_model_call() -> None:
    async def handler(args: dict, ctx: ToolContext) -> str:
        return "screened"

    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "number"},
                        "pregnant": {"type": "boolean"},
                    },
                    "required": ["age"],
                }
            },
            "required": ["profile"],
        },
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
    source = Tool(
        name="screen",
        description="Screen a resident profile",
        parameters={
            "type": "object",
            "properties": {"profile": {"type": "object"}},
            "required": ["profile"],
        },
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
