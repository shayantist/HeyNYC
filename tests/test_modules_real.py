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
    # The generic binding mirrors the active-center layer used by the current Cool Options app.
    # The custom tool also queries the separate year-round Cool Options layer.
    assert binding.source == "arcgis"
    assert "services5.arcgis.com/tMsas0Edz7Aih7fO" in binding.url
    assert "Cooling_Centers_PROD_view" in binding.url
    assert binding.record_id_field == "NYCEM_ID"
    # field_map must cover the keys geo.normalize relies on
    for key in ("name", "lat", "lon", "status"):
        assert key in binding.field_map

    cool_option = bindings["cool_option"]
    assert "services6.arcgis.com/yG5s3afENB5iO9fj" in cool_option.url
    assert "Cool_Options" in cool_option.url
    module = next(module for module in registry.modules if module.name == "cooling_centers")
    assert "services5.arcgis.com" in module.allowlist
    assert "services6.arcgis.com" in module.allowlist


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
