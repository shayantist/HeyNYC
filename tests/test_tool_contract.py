from __future__ import annotations

from jsonschema import Draft202012Validator
from pydantic_ai.models.function import FunctionModel

from heynyc.core import config
from heynyc.core.pydantic_runtime import build_runtime
from heynyc.core.pydantic_runtime.tools import (
    build_module_capabilities,
    resident_fact_confirmation_tool,
)
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
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
    },
    "childcare": {"find_child_care_connect_programs"},
    "clinics": {"find_clinics", "get_health_coverage_guidance"},
    "cooling_centers": {"find_cool_options"},
    "drinking_fountains": set(),
    "events": {"find_nyc_events"},
    "food_pantries": {"find_foodhelp_locations"},
    "housing": {
        "get_housing_guidance",
        "get_hpd_building_records",
        "get_hpd_litigation_records",
    },
    "housing_connect": {"find_housing_connect_lotteries"},
    "immigration": set(),
    "libraries": set(),
    "nyc311_status": {"check_311_request", "search_311_complaints"},
    "public_restrooms": {"find_public_restrooms"},
    "snap_centers": set(),
    "street_closures": {"find_street_closures"},
    "transit": {"check_mta_elevators"},
    "wic": {"find_wic_sites"},
    "workers": {"get_worker_rights_guidance"},
}


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


def test_model_visible_surface_is_pinned():
    registry = _registry()
    shared, capabilities = build_module_capabilities(
        registry, _runtime_tools(registry)
    )

    assert {tool.name for tool in shared} == EAGER_TOOL_NAMES
    assert {
        capability.id: {tool.name for tool in capability.tools}
        for capability in capabilities
    } == CAPABILITY_TOOLS

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
    } == {"clarification_request", "nonfactual_outcome"}


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
        for name, parameter in schema.get("properties", {}).items():
            if "default" not in parameter:
                continue
            default_errors = list(
                Draft202012Validator(parameter).iter_errors(parameter["default"])
            )
            errors.extend(
                f"{tool.name}/{name}: invalid default {error.message}"
                for error in default_errors
            )

    assert errors == []


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
        "clarification_request": [],
        "nonfactual_outcome": [],
    }


def test_known_schema_contracts_match_handler_behavior():
    tools = _runtime_tools(_registry())

    benefits_limit = tools["search_benefits"]._input_schema()["properties"]["limit"]
    assert benefits_limit == {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 8,
        "description": "Maximum benefit programs to return",
    }

    health = tools["get_health_coverage_guidance"]
    assert health._input_schema()["properties"]["topic"]["enum"] == [
        "emergency_care",
        "nyc_care",
        "emergency_medicaid",
        "public_charge",
    ]
    assert "four high-stakes health coverage situations" in health.description

    housing = tools["get_housing_guidance"]
    assert housing._input_schema()["properties"]["topic"]["enum"] == [
        "right_to_counsel",
        "bronx_housing_court",
        "no_water",
        "no_heat",
        "shelter",
        "source_of_income",
    ]
    assert "six high-stakes housing situations" in housing.description

    street_closures = tools["find_street_closures"]._input_schema()
    assert street_closures["required"] == ["near"]


def test_foodhelp_service_window_is_one_complete_input():
    schema = _runtime_tools(_registry())["find_foodhelp_locations"]._input_schema()
    assert "service_window_start" not in schema["properties"]
    assert "service_window_end" not in schema["properties"]
    window = schema["properties"]["service_window"]
    assert window["required"] == ["start", "end"]
    assert window["additionalProperties"] is False

    validator = Draft202012Validator(schema)
    assert not list(
        validator.iter_errors(
            {
                "near": "Union Square",
                "service_window": {"start": "17:00", "end": "23:59"},
            }
        )
    )
    assert list(
        validator.iter_errors(
            {"near": "Union Square", "service_window": {"start": "17:00"}}
        )
    )


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
