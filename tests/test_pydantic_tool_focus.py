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
