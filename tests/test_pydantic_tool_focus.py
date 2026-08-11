from pathlib import Path
from types import SimpleNamespace

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
        ServiceModule(name="benefits", focus_tools=["official_sources"]),
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


def test_hot_water_situation_points_to_parent_that_owns_its_module_tool():
    registry = Registry.discover(Path("heynyc/modules"))
    housing_guidance = _tool("housing_guidance")
    housing_guidance.module = "housing"

    _adapted, capabilities = build_module_capabilities(
        registry,
        {"housing_guidance": housing_guidance},
    )
    capabilities_by_id = {
        capability.id: capability
        for capability in capabilities
    }

    assert "housing-hot-water-code-section" in capabilities_by_id
    parent = capabilities_by_id["housing"]
    situation = capabilities_by_id["housing-hot-water-code-section"]
    assert [tool.name for tool in parent.tools].count("housing_guidance") == 1
    assert all(tool.name != "housing_guidance" for tool in situation.tools)
    assert "load the parent `housing` capability" in "\n".join(
        situation.get_instructions()
    )


def test_active_lockout_uses_native_deferred_capability_selection():
    registry = Registry.discover(Path("heynyc/modules"))
    _adapted, capabilities = build_module_capabilities(registry, {})
    capability = next(
        item for item in capabilities if item.id == "housing-active-lockout"
    )
    assert capability.defer_loading is True
    assert capability.id == "housing-active-lockout"
