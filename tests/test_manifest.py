from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from heynyc.core.manifest import ServiceModule


def test_from_manifest_parses_all_fields(modules_dir: Path):
    module = ServiceModule.from_manifest(modules_dir / "cooling_centers" / "manifest.yaml")
    assert module.name == "cooling_centers"
    assert module.category == "health"
    assert "cooling" in module.keywords
    assert module.datasets[0].id == "h2bn-gu9k"
    assert module.datasets[0].category == "cooling_center"
    assert module.datasets[0].field_map["lat"] == "latitude"
    assert module.datasets[0].where == "status='activated'"
    assert module.seeds == ["https://www.nyc.gov/cooling"]
    assert module.allowlist == ["finder.nyc.gov"]
    assert "geo.nearest" in module.prompt
    # loader populates path, excluded from serialization
    assert module.path == modules_dir / "cooling_centers"
    assert "path" not in module.model_dump()


def test_defaults_for_minimal_manifest(tmp_path: Path):
    p = tmp_path / "manifest.yaml"
    p.write_text("name: minimal\n")
    module = ServiceModule.from_manifest(p)
    assert module.name == "minimal"
    assert module.category == "general"
    assert module.datasets == []
    assert module.seeds == []


def test_unknown_field_rejected(tmp_path: Path):
    p = tmp_path / "manifest.yaml"
    p.write_text("name: bad\ntypo_field: oops\n")
    with pytest.raises(ValidationError):
        ServiceModule.from_manifest(p)


def test_manifest_parses_source_tiers_and_submodule_fields(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "name: events\n"
        "category: events\n"
        "ticketmaster_keyword: world cup\n"
        "source_tiers:\n"
        "  authoritative: [nyctourism.com, nycgovparks.org]\n"
        "  editorial: [timeout.com]\n"
        "  community: [eventbrite.com]\n"
    )
    module = ServiceModule.from_manifest(manifest)
    assert module.source_tiers["authoritative"] == ["nyctourism.com", "nycgovparks.org"]
    assert module.source_tiers["community"] == ["eventbrite.com"]
    assert module.ticketmaster_keyword == "world cup"
    assert module.parent is None  # populated by the registry loader, not YAML
