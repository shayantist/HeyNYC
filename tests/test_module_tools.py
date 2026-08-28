"""Module-specific tools: loading from tools.py and assembling the toolbox."""
from __future__ import annotations

from heynyc.core import config
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox


def test_build_toolbox_uses_shared_web_tools_for_events():
    registry = Registry.discover(config.MODULES_DIR)
    tools = build_toolbox(registry)
    assert {"geocode", "nearest", "distance", "web_search", "web_fetch"} <= set(tools)
    assert "find_nyc_events" not in tools


def test_build_toolbox_keeps_the_local_index_out_of_the_model_tool_surface():
    class Index:
        def search(self, _query, k=5):
            return []

    tools = build_toolbox(Registry.discover(config.MODULES_DIR), index=Index())

    assert "web_search" in tools
    assert "index_search" not in tools
    assert "search_official_guidance" not in tools


def test_toolbox_exposes_only_the_refactored_search_fetch_and_service_names(monkeypatch):
    monkeypatch.setattr(config, "screening_creds", lambda: ("", "", ""))
    tools = build_toolbox(Registry.discover(config.MODULES_DIR))

    assert {
        "web_search",
        "web_fetch",
        "check_notify_nyc",
        "find_cool_options",
        "find_child_care_connect_programs",
        "find_foodhelp_locations",
        "find_housing_connect_lotteries",
    } <= set(tools)
    assert "screen_access_nyc_eligibility" not in tools
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


def test_events_module_exposes_only_the_evidence_extractor():
    registry = Registry.discover(config.MODULES_DIR)
    events = next((m for m in registry.modules if m.name == "events"), None)
    assert events is not None
    assert "authoritative" in events.source_tiers
    assert events.tools == "extraction.py"
    tool_names = {t.name for t in registry.load_module_tools()}
    assert "extract_events" in tool_names
    assert "find_nyc_events" not in tool_names


def test_events_eval_cases_load():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover(config.MODULES_DIR))
    assert any(c.module == "events" for c in cases)


def test_world_cup_is_submodule_of_events():
    registry = Registry.discover(config.MODULES_DIR)
    wc = next((m for m in registry.modules if m.name == "world_cup"), None)
    assert wc is not None
    assert wc.parent == "events"
    assert "ticketmaster_keyword" not in type(wc).model_fields
    # Submodule reuses the parent's shared web-research capability.
    assert wc.tools is None
    # Its seeds are aggregated for index-build:
    assert any("nynjfwc26.com" in s for s in registry.seeds())


def test_world_cup_eval_cases_load_under_submodule():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover(config.MODULES_DIR))
    assert any(c.module == "world_cup" for c in cases)
