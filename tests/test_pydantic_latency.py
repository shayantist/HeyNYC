from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import BaseModel
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from heynyc.core.agent import AgentResult
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule, SituationHint
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import GroundedAnswer
from heynyc.core.pydantic_runtime.runtime import (
    _OutputCorrectionCapability,
    _validation_citation_ids,
)
from heynyc.core.pydantic_runtime.safety import CrisisScreenRun, ScopeScreenRun
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


class _CitedResult(BaseModel):
    primary_citation_id: str
    url: str


def _scope(*modules: str, situations: tuple[str, ...] = ()) -> ScopeScreenRun:
    return ScopeScreenRun(
        event_turn=None,
        modules=modules,
        situations=situations,
        model="scope-test",
        input_tokens=1,
        output_tokens=1,
        cached_input_tokens=0,
        requests=1,
        cost_usd=0.0,
        latency_ms=1,
    )


def _safety() -> CrisisScreenRun:
    return CrisisScreenRun(
        risk="none",
        language="en",
        model="safety-test",
        input_tokens=1,
        output_tokens=1,
        cached_input_tokens=0,
        requests=1,
        cost_usd=0.0,
        latency_ms=1,
    )


async def test_high_stakes_turn_only_offers_grounded_factual_output() -> None:
    ctx = SimpleNamespace(deps=SimpleNamespace(
        current_turn_high_stakes=True,
        validation_rejections=[],
    ), messages=[])
    tools = [
        ToolDefinition(name="final_answer", parameters_json_schema={}),
        ToolDefinition(
            name="grounded_answer",
            parameters_json_schema=GroundedAnswer.model_json_schema(),
        ),
    ]

    prepared = await _OutputCorrectionCapability().prepare_output_tools(ctx, tools)

    assert [tool.name for tool in prepared] == ["grounded_answer"]


def test_grounded_answer_does_not_reject_useful_claim_count() -> None:
    answer = GroundedAnswer.model_validate({
        "grounded_blocks": [
            {"kind": "claim", "text": f"Fact {index}", "citation_ids": ["S1"]}
            for index in range(13)
        ]
    })

    assert len(answer.grounded_blocks) == 13


def test_grounded_block_schema_tells_model_not_to_copy_urls() -> None:
    description = GroundedAnswer.model_json_schema()["$defs"]["GroundedBlock"][
        "properties"
    ]["text"]["description"]

    assert "Do not copy URLs" in description


async def test_output_guard_recovers_only_sources_returned_by_current_tools() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://notify.nyc.gov/old-alert",
        title="Earlier alert",
        snippet="An unrelated source from earlier in the turn.",
    )
    current_id = citations.register(
        "https://www.nyc.gov/site/hpd/help.page",
        title="Current tenant source",
        snippet="Current tenant help.",
    )
    deps = ToolContext(citations=citations, registry=Registry([]))
    deps.tool_result_citation_ids.add(current_id)
    result = AgentResult(
        text="The generated wording was blocked.",
        citations=citations.mapping(),
    )
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([TextPart("unused")])),
        registry=Registry([]),
        tools={},
        output_guard=lambda _text: asyncio.sleep(0, result=frozenset({"unsafe"})),
        guard_grounding=False,
    )

    await runtime._apply_output_guard(result, deps, "en")

    assert "https://www.nyc.gov/site/hpd/help.page" in result.text
    assert "https://notify.nyc.gov/old-alert" not in result.text


def test_validation_recovery_excludes_prior_ordinary_sources() -> None:
    citations = CitationRegistry()
    old_id = citations.register(
        "https://old.example/unrelated",
        title="Prior turn evidence",
        snippet="Unrelated prior evidence.",
    )
    current_id = citations.register(
        "https://current.example/tenant-help",
        title="Current tenant evidence",
        snippet="Current tenant evidence.",
    )

    recovered = _validation_citation_ids(
        [{"citation_ids": [old_id, current_id]}],
        citations,
        {current_id},
    )

    assert recovered == {current_id}


async def test_scope_selected_tools_are_available_and_run_together() -> None:
    started: set[str] = set()
    both_started = asyncio.Event()

    async def handler(args: dict, _ctx: ToolContext) -> str:
        started.add(args["source"])
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return args["source"]

    schema = {
        "type": "object",
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
    }
    tools = {
        name: Tool(
            name=name,
            description=f"Read {name}",
            parameters=schema,
            handler=handler,
            module="housing",
        )
        for name in ("search_311", "get_hpd")
    }
    calls = 0

    async def model(_messages, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        definitions = {tool.name: tool for tool in info.function_tools}
        if calls == 1:
            assert definitions["search_311"].defer_loading is False
            assert definitions["get_hpd"].defer_loading is False
            return ModelResponse(
                [
                    ToolCallPart("search_311", {"source": "311"}, "311-1"),
                    ToolCallPart("get_hpd", {"source": "hpd"}, "hpd-1"),
                ]
            )
        return ModelResponse([TextPart("Done")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([ServiceModule(name="housing", prompt="Use housing data")]),
        tools=tools,
        use_module_capabilities=True,
        scope_screen=lambda _turns: asyncio.sleep(0, result=_scope("housing")),
        guard_grounding=False,
    ).run("Check both records")

    assert result.text == "Done"
    assert started == {"311", "hpd"}
    assert "load_capability" not in result.tool_calls_made
    assert result.usage["capabilities_used"] == ["housing"]
    assert result.usage["model_loaded_capabilities"] == []


async def test_scope_selected_situation_activates_focus_tool_owners() -> None:
    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        return "record"

    calls = 0

    async def model(_messages, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls <= 2:
            definitions = {tool.name: tool for tool in info.function_tools}
            assert definitions["search_311"].defer_loading is False
        if calls == 1:
            return ModelResponse([
                ToolCallPart(
                    "load_capability",
                    {"id": "housing-chronic-repairs"},
                    "redundant-load",
                )
            ])
        return ModelResponse([TextPart("Done")])

    await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([
            ServiceModule(
                name="housing",
                situations=[SituationHint(
                    name="chronic_repairs",
                    definition="Repeated building repairs",
                    focus_tools=["search_311"],
                )],
            ),
            ServiceModule(name="nyc311_status"),
        ]),
        tools={
            "search_311": Tool(
                name="search_311",
                description="Search 311",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
                module="nyc311_status",
            )
        },
        use_module_capabilities=True,
        scope_screen=lambda _turns: asyncio.sleep(
            0,
            result=_scope("housing", situations=("chronic_repairs",)),
        ),
        guard_grounding=False,
    ).run("Check repeated repairs")


async def test_redundant_load_of_preactivated_capability_is_a_noop() -> None:
    executions = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal executions
        executions += 1
        return "record"

    calls = 0

    async def model(_messages, _info) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            return ModelResponse([
                ToolCallPart(
                    "load_capability",
                    {"id": "housing"},
                    f"load-housing-{calls}",
                )
            ])
        if calls == 3:
            return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])
        return ModelResponse([TextPart("Done")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([
            ServiceModule(name="housing", prompt="Use housing data"),
            ServiceModule(name="benefits", prompt="Use benefits data"),
        ]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a record",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
                module="housing",
            )
        },
        use_module_capabilities=True,
        scope_screen=lambda _turns: asyncio.sleep(0, result=_scope("housing")),
        guard_grounding=False,
    ).run("Check housing")

    assert result.status == "success"
    assert result.text == "Done"
    assert executions == 1


async def test_crisis_and_scope_screens_start_concurrently() -> None:
    safety_started = asyncio.Event()
    scope_started = asyncio.Event()
    release = asyncio.Event()

    async def safety_screen(_turns):
        safety_started.set()
        await release.wait()
        return _safety()

    async def scope_screen(_turns):
        scope_started.set()
        await release.wait()
        return _scope()

    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([TextPart("Done")])),
        registry=Registry([]),
        tools={},
        crisis_screen=safety_screen,
        scope_screen=scope_screen,
        guard_grounding=False,
    )
    run = asyncio.create_task(runtime.run("Hello"))

    await asyncio.wait_for(
        asyncio.gather(safety_started.wait(), scope_started.wait()), timeout=1
    )
    release.set()
    result = await run

    assert result.text == "Done"


async def test_exact_read_only_tool_call_is_reused_within_turn() -> None:
    executions = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal executions
        executions += 1
        return "record"

    calls = 0

    async def model(_messages, _info) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            return ModelResponse(
                [ToolCallPart("lookup", {"query": "rats"}, f"lookup-{calls}")]
            )
        return ModelResponse([TextPart("Done")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a record",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lookup,
            )
        },
        guard_grounding=False,
    ).run("Look twice")

    assert result.text == "Done"
    assert executions == 1
    assert result.usage["tool_runs"][-1]["reused"] is True


async def test_concurrent_exact_read_only_calls_share_one_execution() -> None:
    executions = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal executions
        executions += 1
        await asyncio.sleep(0.02)
        return "record"

    calls = 0

    async def model(_messages, _info) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([
                ToolCallPart("lookup", {"query": "rats"}, "lookup-1"),
                ToolCallPart("lookup", {"query": "rats"}, "lookup-2"),
            ])
        return ModelResponse([TextPart("Done")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a record",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lookup,
            )
        },
        guard_grounding=False,
    ).run("Look twice")

    assert executions == 1
    assert sum(run.get("reused", False) for run in result.usage["tool_runs"]) == 1


async def test_different_tool_arguments_are_not_reused() -> None:
    executions = 0

    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        nonlocal executions
        executions += 1
        return "record"

    calls = 0

    async def model(_messages, _info) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            return ModelResponse(
                [ToolCallPart("lookup", {"query": str(calls)}, f"lookup-{calls}")]
            )
        return ModelResponse([TextPart("Done")])

    await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a record",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lookup,
            )
        },
        guard_grounding=False,
    ).run("Look twice")

    assert executions == 2


async def test_cached_typed_result_keeps_citation_ownership() -> None:
    contexts: list[ToolContext] = []

    async def lookup(_args: dict, ctx: ToolContext) -> _CitedResult:
        contexts.append(ctx)
        citation_id = ctx.citations.register(
            "https://nyc.gov/record",
            title="Official record",
            snippet="The record is current.",
        )
        return _CitedResult(
            primary_citation_id=citation_id,
            url="https://nyc.gov/record",
        )

    calls = 0

    async def model(_messages, _info) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{calls}")])
        return ModelResponse([TextPart("Done")])

    await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a cited record",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
                return_type=_CitedResult,
            )
        },
        guard_grounding=False,
    ).run("Look twice")

    assert len(contexts) == 1
    assert contexts[0].required_response_citation_ids == {"S1"}
    assert contexts[0].tool_result_citation_ids == {"S1"}
    assert "https://nyc.gov/record" in contexts[0].tool_result_urls


async def test_preactivated_capability_remains_in_max_turns_telemetry() -> None:
    async def lookup(_args: dict, _ctx: ToolContext) -> str:
        return "record"

    async def model(_messages, _info) -> ModelResponse:
        return ModelResponse([ToolCallPart("lookup", {}, "lookup-1")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([ServiceModule(name="housing", prompt="Use housing data")]),
        tools={
            "lookup": Tool(
                name="lookup",
                description="Read a record",
                parameters={"type": "object", "properties": {}},
                handler=lookup,
                module="housing",
            )
        },
        use_module_capabilities=True,
        scope_screen=lambda _turns: asyncio.sleep(0, result=_scope("housing")),
        usage_limits=UsageLimits(request_limit=1),
        guard_grounding=False,
    ).run("Keep looking")

    assert result.status == "max_turns"
    assert result.usage["capabilities_used"] == ["housing"]
