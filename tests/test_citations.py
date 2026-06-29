from __future__ import annotations

from heynyc.core.citations import CitationRegistry


def test_register_returns_sequential_ids():
    reg = CitationRegistry()
    assert reg.register("https://a.gov", snippet="alpha", kind="DATA") == "S1"
    assert reg.register("https://b.gov", snippet="beta", kind="WEB") == "S2"
    assert len(reg) == 2


def test_identical_source_dedupes():
    reg = CitationRegistry()
    first = reg.register("https://a.gov", snippet="same", kind="DATA")
    second = reg.register("https://a.gov", snippet="same", kind="DATA")
    assert first == second == "S1"
    assert len(reg) == 1


def test_same_url_different_kind_is_distinct():
    reg = CitationRegistry()
    a = reg.register("https://a.gov", snippet="x", kind="DATA")
    b = reg.register("https://a.gov", snippet="x", kind="WEB")
    assert a != b
    assert len(reg) == 2


def test_mapping_shape_and_order():
    reg = CitationRegistry()
    reg.register("https://a.gov", snippet="alpha", title="A", kind="DATA")
    reg.register("https://b.gov", snippet="beta", title="B", kind="WEB")
    mapping = reg.mapping()
    assert list(mapping.keys()) == ["S1", "S2"]
    assert mapping["S1"] == {
        "id": "S1",
        "url": "https://a.gov",
        "title": "A",
        "snippet": "alpha",
        "kind": "DATA",
        "valid_as_of": "",
    }


def test_register_carries_valid_as_of():
    reg = CitationRegistry()
    cid = reg.register(
        "https://access.nyc.gov/snap", snippet="SNAP", title="SNAP",
        kind="DATA", valid_as_of="2026-03-21",
    )
    assert cid == "S1"
    assert reg.mapping()["S1"]["valid_as_of"] == "2026-03-21"


def test_valid_as_of_defaults_empty():
    reg = CitationRegistry()
    reg.register("https://example.gov", kind="WEB")
    assert reg.mapping()["S1"]["valid_as_of"] == ""
