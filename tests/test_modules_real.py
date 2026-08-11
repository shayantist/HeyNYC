"""Drift guard: the real shipped modules under heynyc/modules/ must stay valid."""
from __future__ import annotations

from heynyc.core import config
from heynyc.core.registry import Registry


def test_real_modules_load():
    registry = Registry.discover(config.MODULES_DIR)
    names = {m.name for m in registry.modules}
    assert "cooling_centers" in names


def test_cooling_center_binding_present():
    registry = Registry.discover(config.MODULES_DIR)
    bindings = registry.dataset_bindings()
    assert "cooling_center" in bindings
    binding = bindings["cooling_center"]
    # The finder exposes activated centers and other Cool Options through one current layer.
    assert binding.source == "arcgis"
    assert "services6.arcgis.com/yG5s3afENB5iO9fj" in binding.url
    assert "Cool_Options" in binding.url
    assert binding.where == "Finder_status='OPEN' AND Space_type='Cooling Center'"
    assert binding.record_id_field == "NYCEM_ID"
    # field_map must cover the keys geo.normalize relies on
    for key in ("name", "lat", "lon", "status"):
        assert key in binding.field_map

    module = next(module for module in registry.modules if module.name == "cooling_centers")
    assert "services6.arcgis.com" in module.allowlist
    assert "https://portal.311.nyc.gov/article/?kanumber=KA-02663" in module.seeds


def test_advisories_declares_current_official_notify_cost_sources():
    registry = Registry.discover(config.MODULES_DIR)
    module = next(module for module in registry.modules if module.name == "advisories")

    assert "https://a858-nycnotify.nyc.gov/Home/FAQ" in module.seeds
    assert (
        "https://www.nyc.gov/site/em/resources/notify_nyc/"
        "notify-nyc-short-code-terms-conditions-privacy-policy-information.page"
    ) in module.seeds


def test_index_seeds_exclude_sources_the_target_cannot_ingest():
    registry = Registry.discover(config.MODULES_DIR)
    seeds = {url for module in registry.modules for url in module.seeds}

    blocked = {
        "https://otda.ny.gov/programs/snap/work-requirements.asp",
        "https://otda.ny.gov/policy/directives/2017/",
        "https://otda.ny.gov/hearings/request/",
        "https://otda.ny.gov/oah/",
        "https://home4.nyc.gov/site/cchr/help/residents.page",
        "https://www.nycourts.gov/new-york-city-housing-court/stopping-eviction",
        "https://www.nycourts.gov/new-york-city-housing-court/nyc-housing-court-orders-show-cause",
        "https://www.nycourts.gov/ctapps/Decisions/2026/May26/DecisionList052126.pdf",
        "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/host-cities/new-york-new-jersey",
    }
    assert seeds.isdisjoint(blocked)
    assert (
        "https://inside.fifa.com/tournaments/mens/worldcup/"
        "canadamexicousa2026/media-releases/"
        "fifa-world-cup-26-tm-final-to-be-held-in-new-york-new-jersey-mexico-city-to"
    ) in seeds


def test_public_restroom_binding_uses_the_operational_city_dataset():
    registry = Registry.discover(config.MODULES_DIR)
    binding = registry.dataset_bindings()["public_restroom"]

    assert binding.id == "i7jb-7jku"
    assert binding.where == "status='Operational'"
    assert binding.field_map == {
        "name": "facility_name",
        "lat": "latitude",
        "lon": "longitude",
        "status": "status",
        "website": "website",
    }
    module = next(module for module in registry.modules if module.name == "public_restrooms")
    assert "services6.arcgis.com" in module.allowlist


def test_cooling_cases_cover_archetypes_and_are_valid():
    from heynyc.eval.cases import load_cases

    VALID_TT = {"MFT", "INV", "DIR"}
    VALID_HARM = {"none", "injection", "misinformation", "specialized_advice", "false_premise"}

    cases = [c for c in load_cases(Registry.discover(config.MODULES_DIR)) if c.module == "cooling_centers"]
    ids = {c.id for c in cases}
    assert len(cases) >= 18, "cooling_centers should cover the archetype matrix"
    for c in cases:
        assert c.test_type in VALID_TT, f"{c.id}: bad test_type {c.test_type}"
        assert c.harm_category in VALID_HARM, f"{c.id}: bad harm_category {c.harm_category}"
        if c.test_type == "INV":
            assert c.base in ids, f"{c.id}: INV base '{c.base}' not found"
    assert any(c.harm_category == "injection" for c in cases)
    assert any(c.harm_category == "specialized_advice" for c in cases)
    assert any(c.harm_category == "false_premise" for c in cases)
    assert any(c.test_type == "INV" for c in cases)
    assert any(c.test_type == "DIR" for c in cases)
    assert any(c.invariants.get("must_not_fabricate") for c in cases)


def test_every_module_with_eval_file_exists():
    registry = Registry.discover(config.MODULES_DIR)
    for module in registry.modules:
        if module.eval:
            assert (module.path / module.eval).exists(), f"{module.name} eval file missing"


def test_cross_module_cases_declare_resident_outcomes():
    from heynyc.eval.cases import load_cases

    cases = load_cases(Registry.discover(config.MODULES_DIR))
    by_id = {case.id: case for case in cases}
    expected = {
        "drinking_fountain_cross_module",
        "restroom_open_now",
        "food_holiday_hours",
        "events_groundable_weekend",
        "events_cancelled_not_recommended",
        "adv_false_premise_fee",
        "clinic_immigration_safe_cited",
        "benefits_snap_work_rule_loss_spanish",
    }

    assert all(getattr(by_id[case_id], "utility_criterion", "") for case_id in expected)
    location = by_id["drinking_fountain_cross_module"].utility_criterion.lower()
    assert "directions" in location
    assert "scheduled" in location
    events = by_id["events_cancelled_not_recommended"].utility_criterion.lower()
    assert "already passed" in events
    notify = by_id["adv_false_premise_fee"].utility_criterion.lower()
    assert "generic abstention fails" in notify
    assert "carrier charges" in notify
