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
    # BUG-2 fix: repointed from the wrong Socrata outdoor-misting dataset to the ArcGIS
    # indoor cooling-center finder (NYC Emergency Management), row-addressed by NYCEM_ID.
    assert binding.source == "arcgis"
    assert "CoolingCenters_PROD_view" in binding.url
    assert binding.record_id_field == "NYCEM_ID"
    # field_map must cover the keys geo.normalize relies on
    for key in ("name", "lat", "lon", "status"):
        assert key in binding.field_map


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
