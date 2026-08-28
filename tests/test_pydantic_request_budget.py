from pydantic_ai import UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.manifest import ServiceModule
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def _final_answer_args(answer: str, *citation_ids: str) -> dict:
    del citation_ids
    return {"answer": answer}


async def test_missing_evidence_does_not_force_synthesis_after_six_tool_calls():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Still missing the required fact" if calls < 7 else "Required fact found"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if calls < 7:
            assert "lookup" in {tool.name for tool in info.function_tools}
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("Required fact found"),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Continue looking until the required fact is found",
                handler=lookup,
                idempotent=False,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Find the required fact")

    assert result.status == "success"
    assert result.text == "Required fact found"
    assert calls == 7


async def test_last_allowed_request_remains_an_ordinary_agent_step():
    calls = 0
    requests = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Evidence found"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests < 3:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{requests}")])
        assert info.model_settings.get("tool_choice") != "none"
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("Evidence found"),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up evidence",
                handler=lookup,
                idempotent=False,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=3, tool_calls_limit=3),
    ).run("Find it")

    assert result.status == "success"
    assert result.text == "Evidence found"
    assert calls == 2
    assert requests == 3


async def test_valid_final_answer_skips_a_coemitted_sibling_tool():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "unnecessary"

    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([
            ToolCallPart("lookup", {}, "lookup-1"),
            ToolCallPart(
                "final_answer",
                _final_answer_args("Evidence found"),
                "answer-1",
            ),
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up more evidence",
                handler=lookup,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
    )

    result = await runtime.run("Find it")

    assert result.status == "success"
    assert result.text == "Evidence found"
    assert calls == 0


async def test_model_can_answer_while_function_tools_remain_available():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Verified result"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if calls == 0:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        assert "lookup" in {tool.name for tool in info.function_tools}
        return ModelResponse([TextPart("Verified result")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up a verified result",
                handler=lookup,
            )
        },
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=2),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified result"
    assert calls == 1


async def test_structured_runtime_can_answer_while_tools_remain_available():
    calls = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "No matching record"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if calls == 0:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        assert "lookup" in {tool.name for tool in info.function_tools}
        assert "final_answer" in {tool.name for tool in info.output_tools}
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("No matching record"),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up a current record",
                handler=lookup,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "No matching record"
    assert calls == 1
    assert result.usage["requests"] == 2


async def test_structured_answer_does_not_hide_framework_capability_tools():
    calls = 0
    final_tool_names: set[str] | None = None

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "No matching record"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal final_tool_names
        names = {tool.name for tool in info.function_tools}
        if calls == 0:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls + 1}")])
        final_tool_names = names
        assert "final_answer" in {tool.name for tool in info.output_tools}
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("No matching record"),
                "answer-1",
            )
        ])

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
                handler=lookup,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        use_module_capabilities=True,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert "lookup" in (final_tool_names or set())
    assert "load_capability" in (final_tool_names or set())
    assert "load_capability" not in result.tool_calls_made
    assert calls == 1
    assert result.usage["requests"] == 2


async def test_parallel_tools_leave_the_next_request_free_to_answer():
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
        if requests == 1:
            return ModelResponse([
                ToolCallPart("lookup", {}, f"lookup-{requests}-{index}")
                for index in range(3)
            ])
        assert "lookup" in {tool.name for tool in info.function_tools}
        return ModelResponse([TextPart("Verified partial result")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up verified evidence",
                handler=lookup,
                idempotent=False,
            )
        },
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified partial result"
    assert calls == 3
    assert requests == 2


async def test_parallel_tool_batch_can_be_followed_by_a_structured_answer():
    calls = 0
    requests = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Verified partial result"

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse([
                ToolCallPart("lookup", {}, f"lookup-{calls}-{index}")
                for index in range(4)
            ])
        assert "lookup" in {tool.name for tool in info.function_tools}
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("Verified partial result"),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up verified evidence",
                handler=lookup,
                idempotent=False,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
        usage_limits=UsageLimits(request_limit=10, tool_calls_limit=10),
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified partial result"
    assert calls == 4
    assert requests == 2


async def test_default_request_budget_does_not_reject_a_parallel_tool_batch():
    calls = 0
    requests = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal calls
        calls += 1
        return "Verified partial result"

    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests <= 4:
            return ModelResponse([
                ToolCallPart("lookup", {}, f"lookup-{requests}-{index}")
                for index in range(3)
            ])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args("Verified partial result"),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up verified evidence",
                handler=lookup,
                idempotent=False,
            )
        },
        structured_grounding=True,
        guard_grounding=False,
    ).run("Look it up")

    assert result.status == "success"
    assert result.text == "Verified partial result"
    assert calls == 12
    assert requests == 5


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
        if calls == 0:
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        assert "lookup" in {tool.name for tool in info.function_tools}
        assert "final_answer" in {tool.name for tool in info.output_tools}
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args(
                    "The office is open until 8 PM. {cite:S1}",
                    "S1",
                ),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up official office hours",
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


async def test_reserved_synthesis_rejects_plain_text_tool_syntax():
    requests = 0

    async def lookup(_args: dict, ctx: ToolContext) -> str:
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
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        if requests == 2:
            return ModelResponse([
                TextPart('to=functions.web_fetch {"url":"https://example.com"}')
            ])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                _final_answer_args(
                    "The office is open until 8 PM. {cite:S1}",
                    "S1",
                ),
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Look up official office hours",
                handler=lookup,
            )
        },
        structured_grounding=True,
        usage_limits=UsageLimits(request_limit=3, tool_calls_limit=2),
    ).run("When does the office close?")

    assert result.status == "success"
    assert result.text == "The office is open until 8 PM. {cite:S1}"
    assert "to=functions" not in result.text
    assert requests == 3
