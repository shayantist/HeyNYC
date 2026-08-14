"""Module-specific tools: loading from tools.py and assembling the toolbox."""
from __future__ import annotations

from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox


def test_build_toolbox_includes_module_and_websearch_tools():
    registry = Registry.discover(config.MODULES_DIR)
    tools = build_toolbox(registry)
    # geo + web_search always present; events ships a module tool
    assert {"geocode", "nearest", "distance", "web_search", "find_nyc_events"} <= set(tools)


def test_build_toolbox_keeps_the_local_index_out_of_the_model_tool_surface():
    class Index:
        def search(self, _query, k=5):
            return []

    tools = build_toolbox(Registry.discover(config.MODULES_DIR), index=Index())

    assert "web_search" in tools
    assert "index_search" not in tools
    assert "search_official_guidance" not in tools


def test_toolbox_exposes_only_the_refactored_search_fetch_and_service_names():
    tools = build_toolbox(Registry.discover(config.MODULES_DIR))

    assert {
        "web_search",
        "web_fetch",
        "find_nyc_events",
        "check_notify_nyc",
        "find_cool_options",
        "find_child_care_connect_programs",
        "find_foodhelp_locations",
        "find_housing_connect_lotteries",
        "screen_access_nyc_eligibility",
    } <= set(tools)
    assert {
        "recent_developments",
        "official_sources",
        "whats_on_events",
        "nyc_advisories",
        "cool_options_lookup",
        "nearest_child_care",
        "nearest_food_pantry",
        "open_housing_lotteries",
        "screen_eligibility",
    }.isdisjoint(tools)


def test_events_module_discovered_and_tool_loads():
    registry = Registry.discover(config.MODULES_DIR)
    events = next((m for m in registry.modules if m.name == "events"), None)
    assert events is not None
    assert "authoritative" in events.source_tiers
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "find_nyc_events" in tool_names


def test_events_eval_cases_load():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover(config.MODULES_DIR))
    assert any(c.module == "events" for c in cases)


def test_world_cup_is_submodule_of_events():
    registry = Registry.discover(config.MODULES_DIR)
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

    cases = load_cases(Registry.discover(config.MODULES_DIR))
    assert any(c.module == "world_cup" for c in cases)
