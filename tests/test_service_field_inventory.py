from __future__ import annotations

import importlib.util

from heynyc.core import config

_SCRIPT = config.PROJECT_ROOT / "scripts" / "export_service_field_inventory.py"
_DOCUMENT = config.PROJECT_ROOT / "docs" / "service-location-field-inventory.md"


def _load():
    assert _SCRIPT.exists(), "service-field inventory generator is missing"
    spec = importlib.util.spec_from_file_location(
        "export_service_field_inventory", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inventory_covers_each_provider_and_missing_value_state():
    module = _load()
    inventory = module.inventory()

    assert [source.key for source in inventory] == [
        "foodhelp",
        "wic",
        "childcare",
        "clinics_hrsa",
        "clinics_nyc_care",
    ]
    assert {field.status for source in inventory for field in source.fields} >= {
        "mapped",
        "source_absent",
        "not_applicable",
        "adapter_unmapped",
        "derived",
    }
    assert any(
        source.key == "clinics_nyc_care"
        and field.concept == "service_at_location.facility_type"
        and field.raw_fields == ("hh_type",)
        and field.status == "adapter_unmapped"
        for source in inventory
        for field in source.fields
    )
    assert module.validate_inventory(inventory) == []


def test_inventory_separates_published_schedule_from_request_evaluation():
    module = _load()
    inventory = module.inventory()

    for source in inventory:
        fields = {field.concept: field for field in source.fields}
        assert "schedule.published_hours" in fields
        assert fields["evaluation.distance"].status == "derived"
        assert fields["evaluation.open_now"].status in {"derived", "not_applicable"}
    foodhelp = next(source for source in inventory if source.key == "foodhelp")
    fields = {field.concept: field for field in foodhelp.fields}
    assert fields["schedule.published_hours"].status == "mapped"
    assert fields["evaluation.open_now"].status == "derived"


def test_inventory_matches_adapter_field_ownership_and_fallbacks():
    module = _load()
    sources = {
        source.key: {field.concept: field for field in source.fields}
        for source in module.inventory()
    }

    assert sources["foodhelp"]["service.type"].raw_fields == ("program_type",)
    assert sources["foodhelp"]["service.name"].empty_value == "fallback label"
    assert sources["wic"]["organization.name"].empty_value == "fallback label"
    assert sources["childcare"]["organization.name"].empty_value == "fallback label"
    assert sources["clinics_hrsa"]["organization.name"].empty_value == "fallback label"
    assert sources["childcare"]["service.type"].raw_fields == ("program_type",)
    assert sources["childcare"]["service.type"].invalid_value == "empty label"
    assert sources["childcare"]["service_at_location.facility_type"].raw_fields == (
        "facility_type",
    )
    assert (
        sources["childcare"]["service_at_location.facility_type"].invalid_value
        == "empty label"
    )
    assert sources["clinics_nyc_care"]["organization.name"].empty_value == (
        "fallback label"
    )
    for key in ("foodhelp", "childcare", "clinics_hrsa", "clinics_nyc_care"):
        assert sources[key]["source.record_id"].empty_value == "empty identifier"
    assert sources["foodhelp"]["service_at_location.id"].empty_value == (
        "empty identifier"
    )
    for key in ("foodhelp", "wic"):
        assert sources[key]["source.record_id"].model_path == "service_at_location.id"
    assert sources["wic"]["source.record_id"].empty_value == "derived fallback"
    assert sources["childcare"]["service_at_location.id"].empty_value == (
        "empty identifier"
    )
    for key in ("wic", "clinics_hrsa", "clinics_nyc_care"):
        assert sources[key]["contact.website"].invalid_value == "cleaned text"
    assert sources["clinics_hrsa"]["service.type"].invalid_value == (
        "query-filtered to Active"
    )
    for key in ("clinics_hrsa", "clinics_nyc_care"):
        assert sources[key]["action.url"].raw_fields == ()
        assert sources[key]["action.url"].invalid_value == "deterministic map URL"


def test_generated_inventory_document_has_no_drift():
    module = _load()

    assert _DOCUMENT.read_text() == module.render(module.inventory())
