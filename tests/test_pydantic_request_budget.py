from pydantic_ai import UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.manifest import ServiceModule
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
        messages: list[ModelMessage],
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


async def test_structured_runtime_reserves_synthesis_after_citation_free_tools():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "No matching record"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if info.function_tools:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        assert info.output_tools == []
        return ModelResponse([TextPart("No matching record")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up a current record",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "No matching record"
    assert calls == 8
    assert result.usage["requests"] == 9


async def test_reserved_synthesis_hides_framework_capability_tools():
    calls = 0
    reserved_tool_names: set[str] | None = None

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "No matching record"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal reserved_tool_names
        names = {tool.name for tool in info.function_tools}
        if "lookup" in names:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        if names:
            return ModelResponse([
                ToolCallPart(next(iter(names)), {"id": "events"}, "late-capability")
            ])
        reserved_tool_names = names
        assert info.output_tools == []
        return ModelResponse([TextPart("No matching record")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([
            ServiceModule(
                name="events",
                description="Find current NYC events",
                prompt="Use current evidence",
            )
        ]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up a current record",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        use_module_capabilities=True,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert reserved_tool_names == set()
    assert "load_capability" not in result.tool_calls_made
    assert calls == 8
    assert result.usage["requests"] == 9


async def test_parallel_tools_cannot_consume_the_synthesis_reserve():
    calls = 0
    requests = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Verified partial result"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal requests
        requests += 1
        if info.function_tools:
            return ModelResponse([
                ToolCallPart("lookup", {}, f"lookup-{requests}-{index}")
                for index in range(3)
            ])
        request = next(
            message for message in reversed(messages)
            if isinstance(message, ModelRequest)
        )
        assert "answer now from the useful evidence already retrieved" in (
            request.instructions or ""
        ).lower()
        assert "state the unresolved part as a limitation" in (
            request.instructions or ""
        ).lower()
        return ModelResponse([TextPart("Verified partial result")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up verified evidence",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
        },
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified partial result"
    assert calls == 9
    assert requests == 4


async def test_reserved_synthesis_preserves_authoritative_web_citation():
    calls = 0

    async def lookup(_args: dict, ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="NYC example",
            snippet="The office is open until 8 PM.",
            kind="WEB",
            provenance={
                "evidence_grade": "authoritative",
                "source_tier": "authoritative",
            },
        )
        return f"The office is open until 8 PM. {{cite:{citation_id}}}"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if info.function_tools:
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        assert info.output_tools == []
        return ModelResponse([
            TextPart("The office is open until 8 PM. {cite:S1}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up official office hours",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
            )
        },
        structured_grounding=True,
        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=2),
    ).run("When does the office close?")

    assert result.status == "success"
    assert result.text == "The office is open until 8 PM. {cite:S1}"
    assert result.citations["S1"]["url"] == "https://www.nyc.gov/example"
    assert result.diagnostics["validation_rejections"] == []
    assert calls == 1
