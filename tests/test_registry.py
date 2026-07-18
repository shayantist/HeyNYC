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

    a = ServiceModule(name="events", source_tiers={
        "authoritative": ["NYCTourism.com"], "community": ["eventbrite.com"]})
    b = ServiceModule(name="world_cup", source_tiers={
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
    assert "housing_guidance" in hint.focus_tools
    assert len(hint.definition.split()) >= 8  # meaning, not a keyword
