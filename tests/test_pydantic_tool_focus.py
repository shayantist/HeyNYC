from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from heynyc.core.manifest import ServiceModule
from heynyc.core.pydantic_runtime.tools import build_module_capabilities
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool


async def _handler(_args, _ctx):
    return "ok"


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        handler=_handler,
    )


def test_loaded_module_uses_native_prepare_to_hide_unfocused_shared_tools():
    registry = Registry([
        ServiceModule(
            name="places",
            description="Find a place",
            focus_tools=["nearest"],
        )
    ])
    adapted, _capabilities = build_module_capabilities(
        registry,
        {name: _tool(name) for name in ("nearest", "distance")},
    )
    context = SimpleNamespace(loaded_capability_ids={"places"})

    prepared = {
        tool.name: tool.prepare(
            context,
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters_json_schema={"type": "object", "properties": {}},
            ),
        )
        for tool in adapted
    }

    assert prepared["nearest"] is not None
    assert prepared["distance"] is None


def test_cross_module_capabilities_do_not_narrow_shared_tools():
    registry = Registry([
        ServiceModule(name="places", focus_tools=["nearest"]),
        ServiceModule(name="benefits", focus_tools=["web_fetch"]),
    ])
    adapted, _capabilities = build_module_capabilities(
        registry,
        {name: _tool(name) for name in ("nearest", "distance")},
    )
    context = SimpleNamespace(loaded_capability_ids={"places", "benefits"})

    distance = next(tool for tool in adapted if tool.name == "distance")
    definition = ToolDefinition(
        name="distance",
        description="distance",
        parameters_json_schema={"type": "object", "properties": {}},
    )

    assert distance.prepare(context, definition) is definition


def test_cross_module_no_focus_capability_still_disables_narrowing():
    registry = Registry([
        ServiceModule(name="places", focus_tools=["nearest"]),
        ServiceModule(name="benefits"),
    ])
    adapted, _capabilities = build_module_capabilities(
        registry,
        {name: _tool(name) for name in ("nearest", "distance")},
    )
    context = SimpleNamespace(loaded_capability_ids={"places", "benefits"})
    distance = next(tool for tool in adapted if tool.name == "distance")
    definition = ToolDefinition(
        name="distance",
        description="distance",
        parameters_json_schema={"type": "object", "properties": {}},
    )

    assert distance.prepare(context, definition) is definition


def test_new_turn_capability_hides_stale_module_tools():
    registry = Registry([
        ServiceModule(name="events", description="Find events"),
        ServiceModule(name="food", description="Find food help"),
    ])
    events = _tool("find_events")
    events.module = "events"
    food = _tool("find_food")
    food.module = "food"
    _adapted, capabilities = build_module_capabilities(
        registry,
        {"find_events": events, "find_food": food},
    )
    by_id = {capability.id: capability for capability in capabilities}
    context = SimpleNamespace(
        loaded_capability_ids={"events", "food"},
        messages=[
            ModelRequest(parts=[UserPromptPart("What events are on?")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "events"}, tool_call_id="load-events"
            )]),
            ModelRequest(parts=[UserPromptPart("Now find a food pantry")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "food"}, tool_call_id="load-food"
            )]),
        ],
    )
    definition = lambda name: ToolDefinition(
        name=name,
        description=name,
        parameters_json_schema={"type": "object", "properties": {}},
    )

    event_tool = by_id["events"].tools[0]
    food_tool = by_id["food"].tools[0]
    assert event_tool.prepare(context, definition("find_events")) is None
    assert food_tool.prepare(context, definition("find_food")) is not None


def test_hot_water_situation_is_its_own_deferred_capability():
    registry = Registry.discover(Path("heynyc/modules"))
    housing_guidance = _tool("get_housing_guidance")
    housing_guidance.module = "housing"

    _adapted, capabilities = build_module_capabilities(
        registry,
        {"get_housing_guidance": housing_guidance},
    )
    capabilities_by_id = {
        capability.id: capability
        for capability in capabilities
    }

    assert "housing-hot-water-code-section" in capabilities_by_id
    parent = capabilities_by_id["housing"]
    assert [tool.name for tool in parent.tools].count("get_housing_guidance") == 1
    instructions = "\n".join(
        capabilities_by_id["housing-hot-water-code-section"].get_instructions()
    )
    assert "housing-hot-water-code-section" in instructions
    assert "housing-hot-water-code-section" not in "\n".join(parent.get_instructions())
    assert "load the parent `housing` capability" not in instructions


def test_active_lockout_uses_its_own_deferred_capability():
    registry = Registry.discover(Path("heynyc/modules"))
    _adapted, capabilities = build_module_capabilities(registry, {})
    capability = next(item for item in capabilities if item.id == "housing-active-lockout")
    assert capability.defer_loading is True
    assert "housing-active-lockout" in "\n".join(capability.get_instructions())


def test_situation_catalog_keeps_detail_in_loaded_instructions():
    registry = Registry.discover(Path("heynyc/modules"))
    _adapted, capabilities = build_module_capabilities(registry, {})
    capability = next(
        item for item in capabilities if item.id == "benefits-snap-work-rules"
    )
    instructions = "\n".join(capability.get_instructions())

    assert capability.description.endswith(".")
    assert ". " not in capability.description
    assert capability.description in instructions
    assert len(instructions) > len(capability.description)


def test_cross_module_situation_names_the_capability_that_owns_its_tool():
    registry = Registry.discover(Path("heynyc/modules"))
    food_help = _tool("find_foodhelp_locations")
    food_help.module = "food_pantries"

    _adapted, capabilities = build_module_capabilities(
        registry,
        {"find_foodhelp_locations": food_help},
    )
    capability = next(
        item for item in capabilities if item.id == "benefits-snap-work-rules"
    )
    instructions = "\n".join(capability.get_instructions())

    assert "Load the parent `food_pantries` capability" in instructions
    assert "`find_foodhelp_locations`" in instructions
