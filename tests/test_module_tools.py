"""Module-specific tools: loading from tools.py and assembling the toolbox."""
from __future__ import annotations

from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox


def test_build_toolbox_includes_module_and_websearch_tools():
    registry = Registry.discover(config.MODULES_DIR)
    tools = build_toolbox(registry)
    # geo + web_search always present; events ships a module tool
    assert {"geocode", "nearest", "distance", "web_search", "whats_on_events"} <= set(tools)


def test_events_module_discovered_and_tool_loads():
    registry = Registry.discover()
    events = next((m for m in registry.modules if m.name == "events"), None)
    assert events is not None
    assert "authoritative" in events.source_tiers
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "whats_on_events" in tool_names


def test_events_eval_cases_load():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover())
    assert any(c.module == "events" for c in cases)


def test_world_cup_is_submodule_of_events():
    registry = Registry.discover()
    wc = next((m for m in registry.modules if m.name == "world_cup"), None)
    assert wc is not None
    assert wc.parent == "events"
    assert wc.ticketmaster_keyword == "world cup"
    # Submodule reuses the parent tool — it ships no tools.py of its own.
    assert wc.tools is None
    # Its seeds are aggregated for index-build:
    assert any("nynjfwc26.com" in s for s in registry.seeds())


def test_world_cup_eval_cases_load_under_submodule():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover())
    assert any(c.module == "world_cup" for c in cases)
