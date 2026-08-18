"""Shared fixtures: a temp modules dir with sample manifests."""
from __future__ import annotations

from pathlib import Path

import pytest

from heynyc.core import config
from heynyc.core.registry import Registry


def _write_module(modules_dir: Path, name: str, manifest: str, extra: dict[str, str] | None = None) -> None:
    mod_dir = modules_dir / name
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "manifest.yaml").write_text(manifest)
    for rel, content in (extra or {}).items():
        target = mod_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


@pytest.fixture
def modules_dir(tmp_path: Path) -> Path:
    root = tmp_path / "modules"
    root.mkdir()
    _write_module(
        root,
        "cooling_centers",
        """\
name: cooling_centers
category: health
description: Find cooling centers during heat.
datasets:
  - id: h2bn-gu9k
    category: cooling_center
    field_map: {name: facility, address: address, lat: latitude, lon: longitude}
    where: "status='activated'"
seeds:
  - https://www.nyc.gov/cooling
allowlist:
  - finder.nyc.gov
prompt: |
  Use geo.nearest(category="cooling_center") for heat relief.
""",
    )
    _write_module(
        root,
        "world_cup",
        """\
name: world_cup
category: events
description: World Cup 2026 events and watch parties.
seeds:
  - https://www.nyc.gov/cooling
  - https://nynjfwc26.com/schedule
allowlist:
  - nynjfwc26.com
prompt: |
  Use web_search for watch parties; fan zones are in the module data.
""",
    )
    return root


@pytest.fixture
def registry(modules_dir: Path) -> Registry:
    return Registry.discover(modules_dir, config.BASE_ALLOWLIST)
