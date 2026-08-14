"""Tool package: base abstraction + toolbox assembly."""
from __future__ import annotations

from ..registry import Registry
from .base import Tool, ToolContext, ToolHandler

__all__ = ["Tool", "ToolContext", "ToolHandler", "build_toolbox"]


def build_toolbox(registry: Registry, index=None) -> dict[str, Tool]:
    """Assemble the active tool set.

    Geo and web retrieval tools are always available. The local index is kept
    behind the application boundary rather than exposed as a competing tool.
    The geo tools self-report when a requested category has no dataset.
    """
    from .about import about_tools
    from .geo import geo_tools
    from .web_fetch import web_fetch_tools
    from .web_search import web_search_tools

    tools: dict[str, Tool] = {}
    for tool in geo_tools():
        tools[tool.name] = tool
    for tool in web_search_tools(
        registry.allowlist(), registry.source_tiers(), registry.news_tier()
    ):
        tools[tool.name] = tool
    for tool in web_fetch_tools():
        tools[tool.name] = tool
    for tool in about_tools():
        tools[tool.name] = tool
    # Module-specific tools (the extensibility headline): a module may ship tools.py.
    for tool in registry.load_module_tools():
        tools[tool.name] = tool
    return tools
