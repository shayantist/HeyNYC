"""The LDSS-4826 field map covers what it must and stays the expected shape."""
from __future__ import annotations

from heynyc.modules.benefits.application import SLOTS, load_map


def test_map_loads_and_is_overlay_anchor():
    m = load_map()
    assert m["mode"] == "overlay-anchor"
    assert m["revision"] == "Rev. 12/23"


def test_map_covers_every_required_slot():
    mapped = set(load_map()["fields"])
    required = {s.key for s in SLOTS if s.required}
    missing = required - mapped
    assert not missing, f"form map is missing required slots: {missing}"
