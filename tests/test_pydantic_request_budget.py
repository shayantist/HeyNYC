from pydantic_ai import UsageLimits
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


async def test_last_model_request_is_reserved_for_the_final_answer():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Verified result"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if info.function_tools:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        return ModelResponse([TextPart("Verified result")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up a verified result",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
        },
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=2),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified result"
    assert calls == 1
