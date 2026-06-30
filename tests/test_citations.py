from __future__ import annotations

from heynyc.core.citations import CitationRegistry, api_provenance, content_hash, data_provenance


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
        "provenance": {},
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


def test_data_provenance_hash_is_stable_and_order_independent():
    row_a = {"propertyname": "Marconi Park", "status": "Activated", "y": "40.7"}
    row_b = {"y": "40.7", "status": "Activated", "propertyname": "Marconi Park"}  # reordered
    pa = data_provenance(row_a, record_id="row-1", field_pointer="/status")
    pb = data_provenance(row_b, record_id="row-1", field_pointer="/status")
    assert pa["content_hash"] == pb["content_hash"]                  # key order irrelevant
    assert pa["content_hash"] == content_hash(row_a)
    assert pa["snapshot"] == row_a and pa["record_id"] == "row-1"


def test_register_carries_provenance_through_mapping():
    reg = CitationRegistry()
    prov = data_provenance({"status": "Activated"}, record_id="row-1", field_pointer="/status")
    cid = reg.register("https://data.cityofnewyork.us/resource/x.json?...", kind="DATA",
                       snippet="Marconi Park", provenance=prov)
    assert reg.mapping()[cid]["provenance"]["record_id"] == "row-1"


def test_api_provenance_records_exchange_and_hashes_it():
    prov = api_provenance(
        endpoint="POST https://x/eligibilityPrograms",
        request_summary={"persons": 3, "has_income": True},   # redacted — no raw amounts
        response={"eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]},
        field_pointer="/eligiblePrograms",
        as_of="2026-06-30",
    )
    assert prov["endpoint"].startswith("POST ")
    assert prov["request_summary"] == {"persons": 3, "has_income": True}
    assert prov["field_pointer"] == "/eligiblePrograms" and prov["as_of"] == "2026-06-30"
    assert prov["content_hash"] == content_hash({
        "endpoint": "POST https://x/eligibilityPrograms",
        "request_summary": {"persons": 3, "has_income": True},
        "response": {"eligiblePrograms": [{"code": "S2R007", "name": "SNAP"}]},
    })
