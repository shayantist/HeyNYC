from __future__ import annotations

from pathlib import Path

from heynyc.core import config
from heynyc.core.registry import Registry


def test_discovers_all_modules(registry: Registry):
    names = {m.name for m in registry.modules}
    assert names == {"cooling_centers", "world_cup"}


def test_dataset_bindings_keyed_by_category(registry: Registry):
    bindings = registry.dataset_bindings()
    assert "cooling_center" in bindings
    assert bindings["cooling_center"].id == "h2bn-gu9k"


def test_allowlist_merges_base_and_modules(registry: Registry):
    allow = registry.allowlist()
    for base in config.BASE_ALLOWLIST:
        assert base in allow
    assert "finder.nyc.gov" in allow
    assert "nynjfwc26.com" in allow
    assert allow == sorted(set(allow))  # deduped + sorted


def test_seeds_deduped_order_preserved(registry: Registry):
    seeds = registry.seeds()
    # cooling_centers and world_cup both list the cooling URL; it appears once.
    assert seeds.count("https://www.nyc.gov/cooling") == 1
    assert "https://nynjfwc26.com/schedule" in seeds


def test_capability_blurbs_one_section_per_module(registry: Registry):
    blurbs = registry.capability_blurbs()
    assert "## cooling_centers (health)" in blurbs
    assert "## world_cup (events)" in blurbs


def test_discover_empty_dir(tmp_path: Path):
    reg = Registry.discover(tmp_path / "does_not_exist")
    assert reg.modules == []
    assert reg.seeds() == []


def test_discover_recurses_into_topics(tmp_path: Path):
    mod = tmp_path / "events"
    (mod / "topics" / "world_cup").mkdir(parents=True)
    (mod / "manifest.yaml").write_text("name: events\ncategory: events\nseeds: [https://a.example/x]\n")
    (mod / "topics" / "world_cup" / "manifest.yaml").write_text(
        "name: world_cup\ncategory: events\nseeds: [https://b.example/y]\n"
    )

    registry = Registry.discover(tmp_path)
    names = {m.name for m in registry.modules}
    assert names == {"events", "world_cup"}
    sub = next(m for m in registry.modules if m.name == "world_cup")
    assert sub.parent == "events"
    # aggregators that walk self.modules pick the submodule up for free:
    assert "https://b.example/y" in registry.seeds()


def test_news_tier_kept_separate_from_allowlist():
    # The currency-layer news domains are injected but MUST NOT leak into the trusted allowlist —
    # only the recency check unions them in. (Trust discipline: default search stays gov-grounded.)
    reg = Registry([], base_allowlist=["nyc.gov"], news_tier=["gothamist.com", "nytimes.com"])
    assert reg.news_tier() == ["gothamist.com", "nytimes.com"]
    assert "gothamist.com" not in reg.allowlist()
    assert "nytimes.com" not in reg.allowlist()


def test_source_tiers_aggregates_highest_trust_wins():
    from heynyc.core.manifest import ServiceModule

    a = ServiceModule(name="events", official_only=False, source_tiers={
        "authoritative": ["NYCTourism.com"], "community": ["eventbrite.com"]})
    b = ServiceModule(name="world_cup", official_only=False, source_tiers={
        "editorial": ["eventbrite.com"]})  # same domain, higher tier than community
    tiers = Registry([a, b]).source_tiers()
    assert tiers["nyctourism.com"] == ("authoritative", "events")   # lowercased
    assert tiers["eventbrite.com"][0] == "editorial"                 # editorial(2) beats community(1)


def test_housing_manifest_declares_the_active_lockout_situation():
    """Migration boundary 2: situation hints are module-owned manifest DATA (definition,
    forced-search config, reminder, tool focus), never core constants."""
    from pathlib import Path

    from heynyc.core.registry import Registry

    registry = Registry.discover(Path("heynyc/modules"))
    hints = registry.situation_hints()

    assert "active_lockout" in hints
    module_name, hint = hints["active_lockout"]
    assert module_name == "housing"
    assert hint.high_stakes is True
    assert "lockout" in hint.query
    assert any("311" in url or "nyc.gov" in url for url in hint.urls)
    assert "911" in hint.reminder
    assert "every recommended action" in hint.reminder.lower()
    assert "retrieved official text" in hint.reminder.lower()
    assert "distinct lease and occupancy conditions" in hint.reminder.lower()
    assert "housing_guidance" in hint.focus_tools
    assert "threatened future" in hint.definition
    assert "not" in hint.definition
    assert "recent_developments" not in hint.focus_tools
    assert "web_search" not in hint.focus_tools
    assert "geocode" not in hint.focus_tools
    assert "nearest" not in hint.focus_tools
    assert len(hint.definition.split()) >= 8  # meaning, not a keyword


def test_benefits_manifest_declares_the_snap_work_rules_situation():
    """SNAP work-rules family (ABAWD time limits, work-requirement sanctions, fair-hearing
    recovery) is a benefits-module-owned manifest situation, exactly like active_lockout:
    meaning-based definition, forced-search config, reminder, tool focus, high stakes."""
    from pathlib import Path

    from heynyc.core.registry import Registry

    registry = Registry.discover(Path("heynyc/modules"))
    hints = registry.situation_hints()

    assert "snap_work_rules" in hints
    module_name, hint = hints["snap_work_rules"]
    assert module_name == "benefits"
    assert hint.high_stakes is True
    assert "SNAP" in hint.query and "fair hearing" in hint.query
    assert any(
        host in url
        for url in hint.urls
        for host in ("nyc.gov", "access.nyc.gov", "otda.ny.gov")
    )
    assert "fair-hearing" in hint.reminder
    assert "official_sources" in hint.focus_tools
    # Meaning-based, never a keyword list (the Bengali acid test): the definition must read as a
    # description of the situation, not as SNAP/work terms.
    assert len(hint.definition.split()) >= 8
    lowered = hint.definition.lower()
    assert "food" in lowered and "restore" in lowered


def test_benefits_manifest_declares_mixed_status_snap_situation():
    from heynyc.core.pydantic_runtime.tools import build_module_capabilities

    registry = Registry.discover(Path("heynyc/modules"))
    module_name, hint = registry.situation_hints()["mixed_status_snap"]

    assert module_name == "benefits"
    assert hint.high_stakes is True
    assert "eligible household" in hint.query.lower()
    assert any("snap-application-frequently-asked-questions" in url for url in hint.urls)
    assert any("immigration" in url for url in hint.urls)
    assert any("access.nyc.gov/eligibility" in url for url in hint.urls)
    assert "official access nyc eligibility screener" in hint.reminder.lower()
    assert "unless the resident explicitly requests or accepts" in hint.reminder.lower()
    assert "required outcomes" in hint.reminder.lower()
    assert "immigration-safe legal-help route" in hint.reminder.lower()
    assert "the final sentence must ask" not in hint.reminder.lower()
    assert hint.focus_tools == ["official_sources"]
    _, capabilities = build_module_capabilities(registry, {})
    capability = next(
        item for item in capabilities if item.id == "benefits-mixed-status-snap"
    )
    instructions = "\n".join(capability.get_instructions())
    assert "call `official_sources` with every official pages url" in instructions.lower()


def test_benefits_manifest_declares_generic_snap_denial_situation():
    from heynyc.core.pydantic_runtime.tools import build_module_capabilities

    registry = Registry.discover(Path("heynyc/modules"))
    module_name, hint = registry.situation_hints()["snap_denial_fair_hearing"]

    assert module_name == "benefits"
    assert hint.high_stakes is True
    assert any("LDSS_4826A.pdf" in url for url in hint.urls)
    assert "food assistance is additional" in hint.reminder.lower()
    assert hint.focus_tools == ["official_sources"]
    _, capabilities = build_module_capabilities(registry, {})
    capability = next(
        item for item in capabilities if item.id == "benefits-snap-denial-fair-hearing"
    )
    instructions = "\n".join(capability.get_instructions())
    assert "call `official_sources` with every official pages url" in instructions.lower()


def test_benefits_manifest_declares_medicaid_termination_hearing_situation():
    from heynyc.core.pydantic_runtime.tools import build_module_capabilities

    registry = Registry.discover(Path("heynyc/modules"))
    module_name, hint = registry.situation_hints()["medicaid_termination_fair_hearing"]

    assert module_name == "benefits"
    assert hint.high_stakes is True
    assert any("final_iad_ac" in url for url in hint.urls)
    assert any("final_fad_ac" in url for url in hint.urls)
    assert any("medicaid/how_do_i_apply" in url for url in hint.urls)
    assert "otda decides" in hint.reminder.lower()
    assert "resident's language" in hint.reminder.lower()
    assert hint.focus_tools == ["official_sources"]
    _, capabilities = build_module_capabilities(registry, {})
    capability = next(
        item for item in capabilities if item.id == "benefits-medicaid-termination-fair-hearing"
    )
    instructions = "\n".join(capability.get_instructions()).lower()
    assert "call `official_sources` with every official pages url" in instructions


def test_official_only_is_the_default_and_blocks_editorial_sources_at_load():
    """RULED 2026-07-18: retrieval pools are stakes-tiered, and the stakes declaration lives in
    each module's OWN manifest (`official_only`, default true), enforced by the schema at load —
    a high-stakes module physically cannot grow an editorial pool without an explicit opt-out."""
    import pytest

    from heynyc.core.manifest import ServiceModule

    with pytest.raises(ValueError, match="official_only"):
        ServiceModule(
            name="benefits_like",
            source_tiers={"editorial": ["example.com"]},
        )

    lifestyle = ServiceModule(
        name="events_like",
        official_only=False,
        source_tiers={"editorial": ["example.com"]},
    )
    assert lifestyle.official_only is False


def test_events_editorial_pool_covers_nyc_lifestyle_press():
    """The approved editorial tier for lifestyle discovery: the sources that actually answer
    'where do French fans watch' (F-none: the Google-comparison transcript, 2026-07-18)."""
    registry = Registry.discover(Path("heynyc/modules"))
    events = next(m for m in registry.modules if m.name == "events")
    assert events.official_only is False
    editorial = set(events.source_tiers.get("editorial", ()))
    for domain in ("theinfatuation.com", "eater.com", "cntraveler.com", "bkmag.com"):
        assert domain in editorial
        assert domain in registry.allowlist()


def test_events_keeps_known_source_trust_separate_from_catalog_scope():
    registry = Registry.discover(Path("heynyc/modules"))
    events = next(module for module in registry.modules if module.name == "events")

    assert registry.source_tiers()["nba.com"][0] == "authoritative"
    assert "nba.com" in events.allowlist
    prompt = " ".join(events.prompt.lower().split())
    assert "requests for event choices" in prompt
    assert "web_search remains available" in prompt
    assert "professional-team" not in prompt


def test_libraries_declares_each_nyc_library_system_as_authoritative():
    registry = Registry.discover(Path("heynyc/modules"))
    libraries = next(module for module in registry.modules if module.name == "libraries")

    assert libraries.seeds == []
    assert set(libraries.allowlist) == {
        "bklynlibrary.org",
        "nypl.org",
        "queenslibrary.org",
    }
    for domain in libraries.allowlist:
        assert registry.source_tiers()[domain][0] == "authoritative"
    prompt = " ".join(libraries.prompt.lower().split())
    assert "current branch page" in prompt
    assert "systemwide technology page" in prompt


def test_workers_separates_informal_guidance_from_a_formal_complaint():
    registry = Registry.discover(Path("heynyc/modules"))
    workers = next(module for module in registry.modules if module.name == "workers")
    prompt = " ".join(workers.prompt.lower().split())

    assert "informal inquiry" in prompt
    assert "formal complaint" in prompt
    assert "employer notice" in prompt


def test_transit_manifest_declares_current_mta_accessibility_sources():
    registry = Registry.discover(Path("heynyc/modules"))
    transit = next(module for module in registry.modules if module.name == "transit")

    assert "mta.info" in transit.allowlist
    assert registry.source_tiers()["mta.info"][0] == "authoritative"
    assert any("accessibility" in seed for seed in transit.seeds)
    prompt = " ".join(transit.prompt.lower().split())
    assert "accessible trip" in prompt
    assert "current elevator" in prompt
    assert "route-level accessibility and service status remain unverified" in prompt
    assert "requested day" in prompt
    assert "requires a grounded handoff" in prompt
    assert "ask that exact endpoint question in the grounded handoff" in prompt
    assert "call official_sources directly" in prompt
    assert "do not use web_search" in prompt
    assert "do not suggest a candidate address or entrance" in prompt


def test_childcare_marks_the_official_myschools_service_authoritative():
    registry = Registry.discover(Path("heynyc/modules"))

    assert registry.source_tiers()["myschools.nyc"][0] == "authoritative"


def test_benefits_discovery_does_not_claim_an_unloaded_screening_workflow():
    registry = Registry.discover(Path("heynyc/modules"))
    benefits = next(module for module in registry.modules if module.name == "benefits")

    assert "unless its deferred workflow capability is loaded" in benefits.prompt.lower()


def test_food_help_manifest_handles_citywide_starting_point_before_location():
    registry = Registry.discover(Path("heynyc/modules"))
    food = next(module for module in registry.modules if module.name == "food_pantries")
    prompt = " ".join(food.prompt.lower().split())

    assert "where do i start" in prompt
    assert "official_sources" in prompt
    assert "before optionally asking for a location" in prompt
    assert "exact location phrase" in prompt


def test_governed_screening_capability_explains_estimate_before_intake():
    from heynyc.core.pydantic_runtime.tools import (
        build_module_capabilities,
        resident_fact_confirmation_tool,
    )
    from heynyc.core.tools.base import Tool
    from heynyc.modules.benefits.tools import screen_eligibility_tool

    async def search_handler(args, ctx):
        return "official guidance"

    registry = Registry.discover(Path("heynyc/modules"))
    screening = screen_eligibility_tool()
    screening.module = "benefits"
    confirmation = resident_fact_confirmation_tool(screening)
    confirmation.module = "benefits"
    discovery = Tool(
        name="benefits_search",
        description="Find current benefit programs",
        parameters={"type": "object", "properties": {}},
        handler=search_handler,
        module="benefits",
    )
    _, capabilities = build_module_capabilities(
        registry,
        {
            screening.name: screening,
            confirmation.name: confirmation,
            discovery.name: discovery,
        },
    )
    capability = next(
        item for item in capabilities if item.id == "benefits-screen-eligibility"
    )
    instructions = "\n".join(capability.get_instructions()).lower()

    assert "guaranteed approval" in instructions
    assert "benefits_search" not in capability.get_toolset().tools
    assert "load the parent `benefits` capability" in instructions
    assert "use `search_tools` to discover and call" in instructions
    assert "`benefits_search`" in instructions
    assert "estimate, not a determination" in instructions
