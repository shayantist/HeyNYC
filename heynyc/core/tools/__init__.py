"""Tool package: base abstraction + toolbox assembly."""
from __future__ import annotations

from ..registry import Registry
from .base import Tool, ToolContext, ToolHandler

__all__ = ["Tool", "ToolContext", "ToolHandler", "build_toolbox"]


def build_toolbox(registry: Registry, index=None) -> dict[str, Tool]:
    """Assemble the active tool set.

    Geo tools are always available. index_search is added when an IndexRetriever
    is provided (a built corpus). web_search is layered on in a later phase. The
    geo tools self-report when a requested category has no dataset.
    """
    from .geo import geo_tools
    from .web_search import web_search_tools

    tools: dict[str, Tool] = {}
    for tool in geo_tools():
        tools[tool.name] = tool
    for tool in web_search_tools(registry.allowlist(), registry.source_tiers()):
        tools[tool.name] = tool
    if index is not None:
        from .index_search import index_search_tools

        for tool in index_search_tools(index):
            tools[tool.name] = tool
    # Module-specific tools (the extensibility headline): a module may ship tools.py.
    for tool in registry.load_module_tools():
        tools[tool.name] = tool
    return tools
