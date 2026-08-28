from __future__ import annotations

from jsonschema import Draft202012Validator
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from heynyc.core import config
from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime import build_runtime
from heynyc.core.pydantic_runtime.tools import (
    build_module_capabilities,
    resident_fact_confirmation_tool,
    runtime_tool,
)
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.base import Tool, ToolContext, ToolFailure, ToolInput
from heynyc.modules.benefits.tools import screen_access_nyc_eligibility_tool

EAGER_TOOL_NAMES = {
    "about_heynyc",
    "distance",
    "geocode",
    "nearest",
    "web_fetch",
    "web_search",
}

CAPABILITY_TOOLS = {
    "advisories": {"check_notify_nyc"},
    "benefits": {
        "search_benefits",
        "confirm_screen_access_nyc_eligibility_facts",
        "get_utility_shutoff_guidance",
    },
    "childcare": {"find_child_care_connect_programs"},
    "clinics": {"find_clinics", "get_health_coverage_guidance"},
    "cooling_centers": {"find_cool_options"},
    "drinking_fountains": set(),
    "events": {"extract_events"},
    "food_pantries": {"find_foodhelp_locations"},
    "housing": {
        "get_housing_guidance",
        "get_hpd_building_records",
        "get_hpd_litigation_records",
    },
    "housing_connect": {"find_housing_connect_lotteries"},
    "immigration": set(),
    "libraries": {"find_bpl_branches"},
    "nyc311_status": {"check_311_request", "search_311_complaints"},
    "public_restrooms": {"find_public_restrooms"},
    "snap_centers": set(),
    "street_closures": {"find_street_closures"},
    "transit": {"check_mta_elevators"},
    "wic": {"find_wic_sites"},
    "workers": {"get_worker_rights_guidance"},
}

SITUATION_CAPABILITY_TOOLS = {}


def _registry() -> Registry:
    return Registry.discover(
        config.MODULES_DIR,
        base_allowlist=config.BASE_ALLOWLIST,
        news_tier=config.NEWS_ALLOWLIST,
    )


def _runtime_tools(registry: Registry):
    tools = build_toolbox(registry)
    screening = screen_access_nyc_eligibility_tool()
    screening.module = "benefits"
    tools[screening.name] = screening
    confirmation = resident_fact_confirmation_tool(screening)
    confirmation.module = "benefits"
    tools[confirmation.name] = confirmation
    return tools


def _missing_descriptions(schema: dict, path: str = "") -> list[str]:
    missing: list[str] = []
    if schema.get("type") == "object":
        for name, child in schema.get("properties", {}).items():
            child_path = f"{path}/{name}"
            if not child.get("description"):
                missing.append(child_path)
            missing.extend(_missing_descriptions(child, child_path))
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        item_path = f"{path}/*"
        if not schema["items"].get("description") and "$ref" not in schema["items"]:
            missing.append(item_path)
        missing.extend(_missing_descriptions(schema["items"], item_path))
    return missing


async def test_typed_confirmation_records_scoped_resident_facts() -> None:
    class ProfileInput(ToolInput):
        age: int

    async def lookup(request: ProfileInput, _ctx: ToolContext) -> str:
        return f"age {request.age}"

    source = Tool(
        name="lookup",
        description="Lookup",
        input_type=ProfileInput,
        handler=lookup,
        resident_fact_scope=("/age",),
    )
    confirmation = resident_fact_confirmation_tool(source)
    ctx = ToolContext(
        citations=CitationRegistry(), registry=Registry([]), user_turns=("I am 42",),
    )

    assert await confirmation.invoke({"age": 42}, ctx) == "age 42"
    assert ctx.resident_facts["/age"].value == 42
    assert ctx.resident_facts["/age"].status == "confirmed"


def test_module_confirmation_keeps_native_typed_contract() -> None:
    class ProfileInput(ToolInput):
        age: int
        note: str | None = None

    async def lookup(request: ProfileInput, _ctx: ToolContext) -> str:
        return f"age {request.age}"

    source = Tool(
        name="lookup",
        description="Lookup",
        input_type=ProfileInput,
        handler=lookup,
        resident_fact_scope=("/age",),
        module="benefits",
    )
    confirmation = resident_fact_confirmation_tool(source)
    confirmation.module = "benefits"
    _, capabilities = build_module_capabilities(
        _registry(),
        {source.name: source, confirmation.name: confirmation},
    )
    capability = next(item for item in capabilities if item.id == "benefits")
    adapted = capability.get_toolset().tools[confirmation.name]

    assert adapted.function.__annotations__["request"] is ProfileInput


async def test_adapter_preserves_declared_nested_pydantic_result() -> None:
    class ResultRow(BaseModel):
        name: str
        hours: str | None
        citations: list[str]
        action_url: str

    class ProviderResult(BaseModel):
        records: list[ResultRow]
        next_cursor: str | None
        error: str | None

    async def handler(_args: dict, _ctx: ToolContext) -> dict:
        return {
            "records": [{
                "name": "Example center",
                "hours": None,
                "citations": ["S1"],
                "action_url": "https://www.nyc.gov/example/center",
            }],
            "next_cursor": None,
            "error": None,
        }

    source = Tool(
        name="typed_lookup",
        description="Return one typed provider result",
        handler=handler,
        return_type=ProviderResult,
    )
    seen: list[object] = []

    async def model(messages: list[ModelMessage], _info) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("typed_lookup", {}, "lookup-1")])
        seen.append(returns[-1].content)
        return ModelResponse([TextPart("Done")])

    context = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    await Agent(
        FunctionModel(model),
        deps_type=ToolContext,
        tools=[runtime_tool(source)],
    ).run("Find it", deps=context)

    assert len(seen) == 1
    assert isinstance(seen[0], ProviderResult)
    assert seen[0].model_dump() == {
        "records": [{
            "name": "Example center",
            "hours": None,
            "citations": ["S1"],
            "action_url": "https://www.nyc.gov/example/center",
        }],
        "next_cursor": None,
        "error": None,
    }
    assert context.tool_result_urls == {"https://www.nyc.gov/example/center"}
    assert context.tool_runs[0]["tool"] == "typed_lookup"
    assert context.tool_runs[0]["status"] == "success"
    assert context.tool_runs[0]["latency_ms"] >= 0


def test_tool_failure_is_a_compact_typed_outcome() -> None:
    failure = ToolFailure(
        status="unavailable",
        reason="The page blocked both retrieval methods.",
        retryable=False,
        source_url="https://www.nyc.gov/example",
    )

    assert failure.model_dump(mode="json") == {
        "status": "unavailable",
        "reason": "The page blocked both retrieval methods.",
        "retryable": False,
        "source_url": "https://www.nyc.gov/example",
    }


async def test_web_fetch_registers_only_public_links_found_in_fetched_evidence(
    monkeypatch,
) -> None:
    async def resolve(url: str, *, allow_local: bool):
        assert allow_local is False
        assert url == "https://www.nyc.gov/example/application"

    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.tools.validate_and_resolve_url",
        resolve,
    )

    async def handler(_args: dict, _ctx: ToolContext) -> str:
        return (
            "SOURCE S1: Official page\n"
            "Apply here: https://www.nyc.gov/example/application\n"
            "Ignore this unsafe link: http://127.0.0.1/private"
        )

    source = Tool(
        name="web_fetch",
        description="Fetch one official page",
        handler=handler,
    )

    async def model(messages: list[ModelMessage], _info) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("web_fetch", {}, "fetch-1")])
        return ModelResponse([TextPart("Done")])

    context = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    await Agent(
        FunctionModel(model),
        deps_type=ToolContext,
        tools=[runtime_tool(source)],
    ).run("Find the official application", deps=context)

    assert context.tool_result_urls == {"https://www.nyc.gov/example/application"}


async def test_adapter_reports_invalid_declared_result_as_terminal_tool_failure() -> None:
    class ProviderResult(BaseModel):
        count: int

    async def handler(_args: dict, _ctx: ToolContext) -> dict:
        return {"count": {"invalid": True}}

    source = Tool(
        name="typed_lookup",
        description="Return one typed provider result",
        handler=handler,
        return_type=ProviderResult,
    )
    failures: list[str] = []

    async def model(messages: list[ModelMessage], _info) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("typed_lookup", {}, "lookup-1")])
        failures.append(str(returns[-1].content))
        return ModelResponse([TextPart("The provider result was unavailable.")])

    context = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    result = await Agent(
        FunctionModel(model),
        deps_type=ToolContext,
        tools=[runtime_tool(source)],
    ).run("Find it", deps=context)

    assert result.output == "The provider result was unavailable."
    assert failures == ["typed_lookup returned an invalid structured result"]
    assert context.tool_runs[0]["status"] == "error"
    assert context.tool_runs[0]["error"] == "ValidationError"


async def test_adapter_rejects_invalid_structured_action_url_before_model_use() -> None:
    class ProviderResult(BaseModel):
        action_url: str

    async def handler(_args: dict, _ctx: ToolContext) -> ProviderResult:
        return ProviderResult(action_url="http://example.com/unsafe")

    source = Tool(
        name="typed_lookup",
        description="Return one typed provider result",
        handler=handler,
        return_type=ProviderResult,
    )
    failures: list[str] = []

    async def model(messages: list[ModelMessage], _info) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("typed_lookup", {}, "lookup-1")])
        failures.append(str(returns[-1].content))
        return ModelResponse([TextPart("The action link was unavailable.")])

    context = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    result = await Agent(
        FunctionModel(model),
        deps_type=ToolContext,
        tools=[runtime_tool(source)],
    ).run(
        "Find it",
        deps=context,
    )
    assert context.tool_runs[0]["status"] == "error"
    assert context.tool_runs[0]["error"] == "ToolFailed"

    assert result.output == "The action link was unavailable."
    assert failures == ["typed_lookup returned invalid action metadata"]


def test_model_visible_surface_is_pinned():
    registry = _registry()
    shared, capabilities = build_module_capabilities(
        registry, _runtime_tools(registry)
    )

    assert {tool.name for tool in shared} == EAGER_TOOL_NAMES
    actual_capabilities = {
        capability.id: {tool.name for tool in capability.tools}
        for capability in capabilities
    }
    expected_capabilities = dict(CAPABILITY_TOOLS)
    for module in registry.modules:
        root = module.parent or module.name
        module_prefix = root.replace("_", "-")
        for hint in module.situations:
            hint_id = hint.name.replace("_", "-").removeprefix(
                f"{module_prefix}-"
            )
            capability_id = f"{module_prefix}-{hint_id}"
            expected_capabilities[capability_id] = SITUATION_CAPABILITY_TOOLS.get(
                capability_id, set()
            )
    assert actual_capabilities == expected_capabilities
    tool_owners: dict[str, str] = {}
    for capability in capabilities:
        for tool in capability.tools:
            assert tool.name not in tool_owners, (
                f"{tool.name} is duplicated by {tool_owners[tool.name]} and {capability.id}"
            )
            tool_owners[tool.name] = capability.id
    utility = next(
        capability for capability in capabilities
        if capability.id == "benefits-utility-shutoff"
    )
    utility_instructions = "\n".join(utility._instructions)
    assert (
        "Use the parent `benefits` capability tools: "
        "`get_utility_shutoff_guidance`"
    ) in utility_instructions

    runtime = build_runtime(
        registry,
        model=FunctionModel(lambda *_args: None),
        tools=_runtime_tools(registry),
        use_module_capabilities=True,
        structured_grounding=True,
    )
    assert {
        definition.name
        for definition in runtime._agent._output_schema.toolset._tool_defs
    } == {
        "final_answer",
        "grounded_answer",
        "clarification_request",
        "nonfactual_outcome",
    }


def test_tool_parameters_are_described_and_self_consistent():
    errors: list[str] = []
    for tool in _runtime_tools(_registry()).values():
        schema = tool._input_schema()
        if schema.get("additionalProperties") is not False:
            errors.append(f"{tool.name}: permits unknown properties")
        errors.extend(
            f"{tool.name}{path}: missing description"
            for path in _missing_descriptions(schema)
        )
        validator = Draft202012Validator(schema)
        for name, parameter in schema.get("properties", {}).items():
            if "default" not in parameter:
                continue
            default_errors = list(
                validator.descend(parameter["default"], parameter, schema_path=name)
            )
            errors.extend(
                f"{tool.name}/{name}: invalid default {error.message}"
                for error in default_errors
            )

    assert errors == []


def test_model_facing_tools_expose_resident_constraints_not_runtime_controls():
    tools = _runtime_tools(_registry())

    assert "find_nyc_events" not in tools
    assert set(tools["geocode"]._input_schema()["properties"]) == {"text"}
    assert set(tools["web_fetch"]._input_schema()["properties"]) == {
        "url",
        "find",
    }
    assert set(tools["web_search"]._input_schema()["properties"]) == {
        "queries",
        "domains",
        "published_after",
        "published_before",
    }
    assert set(tools["find_foodhelp_locations"]._input_schema()["properties"]) == {
        "near",
        "max_results",
        "starts_after",
        "starts_before",
        "active_at",
        "site",
        "service_type",
    }
    assert set(tools["find_cool_options"]._input_schema()["properties"]) == {
        "near",
        "max_results",
        "active_at",
        "site",
        "exclude_sites",
        "kind",
        "audience",
    }
    assert set(tools["search_benefits"]._input_schema()["properties"]) == {
        "query",
        "max_results",
    }
    assert set(tools["check_notify_nyc"]._input_schema()["properties"]) == set()
    assert "list_notify_nyc" not in tools


def test_tools_leave_provider_strictness_unset_by_default():
    tools = _runtime_tools(_registry())

    assert all(tool.strict is None for tool in tools.values())
    assert all(runtime_tool(tool).strict is None for tool in tools.values())


def test_output_tool_parameters_are_described():
    registry = _registry()
    runtime = build_runtime(
        registry,
        model=FunctionModel(lambda *_args: None),
        tools=_runtime_tools(registry),
        use_module_capabilities=True,
        structured_grounding=True,
    )

    assert {
        definition.name: _missing_descriptions(definition.parameters_json_schema)
        for definition in runtime._agent._output_schema.toolset._tool_defs
    } == {
        "final_answer": [],
        "grounded_answer": [],
        "clarification_request": [],
        "nonfactual_outcome": [],
    }


def test_final_answer_tool_is_one_concise_terminal_prose_contract():
    runtime = build_runtime(
        _registry(),
        model=FunctionModel(lambda *_args: None),
        tools=_runtime_tools(_registry()),
        use_module_capabilities=True,
        structured_grounding=True,
    )
    final_answer = next(
        definition
        for definition in runtime._agent._output_schema.toolset._tool_defs
        if definition.name == "final_answer"
    )

    assert "complete final answer" in final_answer.description.lower()
    assert "not a status update" in final_answer.description.lower()
    assert len(final_answer.description.split()) <= 55
    assert set(final_answer.parameters_json_schema["properties"]) == {"answer"}
    assert final_answer.parameters_json_schema["required"] == ["answer"]
    answer_description = final_answer.parameters_json_schema["properties"]["answer"][
        "description"
    ].lower()
    assert "resident-facing prose with inline citations" in answer_description
    assert len(answer_description.split()) <= 35


def test_known_schema_contracts_match_handler_behavior():
    tools = _runtime_tools(_registry())

    benefits_limit = tools["search_benefits"]._input_schema()["properties"]["max_results"]
    assert benefits_limit["anyOf"][0] == {"minimum": 1, "type": "integer"}
    assert benefits_limit["default"] is None
    assert "resident-requested program count" in benefits_limit["description"].lower()

    health = tools["get_health_coverage_guidance"]
    assert health._input_schema()["properties"]["topic"]["enum"] == [
        "emergency_care",
        "nyc_care",
        "emergency_medicaid",
        "public_charge",
    ]
    assert "four high-stakes health coverage situations" in health.description

    housing = tools["get_housing_guidance"]
    assert set(housing._input_schema()["properties"]["topic"]["enum"]) == {
        "right_to_counsel",
        "bronx_housing_court",
        "no_water",
        "no_heat",
        "shelter",
        "source_of_income",
    }
    assert "six high-stakes housing situations" in housing.description

    street_closures = tools["find_street_closures"]._input_schema()
    assert street_closures["required"] == ["near"]


def test_foodhelp_uses_one_absolute_service_window():
    schema = _runtime_tools(_registry())["find_foodhelp_locations"]._input_schema()
    assert "service_window" not in schema["properties"]
    assert {"starts_after", "starts_before", "active_at"} <= set(schema["properties"])


def test_311_operations_do_not_accept_each_others_inputs():
    tools = _runtime_tools(_registry())
    status = tools["check_311_request"]._input_schema()
    complaints = tools["search_311_complaints"]._input_schema()

    assert status["required"] == ["sr_number"]
    assert list(Draft202012Validator(status).iter_errors({"about": "noise"}))
    assert list(
        Draft202012Validator(complaints).iter_errors({"sr_number": "69741503"})
    )


def test_web_search_description_owns_operation_not_global_answer_policy():
    description = _runtime_tools(_registry())["web_search"].description.lower()

    assert "short noun-phrase query" in description
    assert "authoritative excerpts" in description
    assert "contested legal matter" not in description
    assert "lead with the protection" not in description
    assert "court" not in description
