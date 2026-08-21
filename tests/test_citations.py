from __future__ import annotations

from heynyc.core import citations
from heynyc.core.agent import _urls_in
from heynyc.core.citations import (
    Citation,
    CitationRegistry,
    api_provenance,
    content_hash,
    data_provenance,
)


def test_highlight_url_is_available():
    assert hasattr(citations, "text_fragment_url")


def test_urls_in_ignores_markdown_emphasis_delimiters():
    assert _urls_in("**https://otda.ny.gov/hearings/**") == {
        "https://otda.ny.gov/hearings"
    }


def test_urls_in_keeps_an_internal_asterisk() -> None:
    assert _urls_in("See https://example.org/a*b for details") == {
        "https://example.org/a*b"
    }


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


def test_reused_source_is_marked_as_touched_in_the_current_turn() -> None:
    reg = CitationRegistry()
    citation_id = reg.register("https://a.gov", snippet="same", kind="WEB")
    reg.begin_turn()

    assert reg.touched_ids() == set()
    assert reg.register("https://a.gov", snippet="same", kind="WEB") == citation_id
    assert reg.touched_ids() == {citation_id}


def test_touch_cursor_includes_a_reused_citation() -> None:
    reg = CitationRegistry()
    citation_id = reg.register("https://a.gov", snippet="same", kind="WEB")
    cursor = reg.touch_cursor()

    reg.register("https://a.gov", snippet="same", kind="WEB")

    assert reg.touched_since(cursor) == {citation_id}


def test_same_url_with_different_evidence_does_not_reuse_stale_citation():
    reg = CitationRegistry()
    prefix = "same opening " * 12

    first = reg.register("https://a.gov", snippet=prefix + "old detail", kind="WEB")
    second = reg.register("https://a.gov", snippet=prefix + "new detail", kind="WEB")

    assert first != second
    assert reg.mapping()[second]["snippet"].endswith("new detail")


def test_f197_same_data_summary_with_a_new_snapshot_gets_a_new_citation() -> None:
    reg = CitationRegistry()
    snippet = "5 nearby sites had weekly hours"
    sunset = data_provenance(
        {"origin_label": "Sunset Park", "scheduled_open_nearby": 5},
        record_id="availability-summary",
        field_pointer="/",
    )
    queens = data_provenance(
        {"origin_label": "82nd Street and Roosevelt Avenue", "scheduled_open_nearby": 5},
        record_id="availability-summary",
        field_pointer="/",
    )

    first = reg.register("https://data.nyc.gov/food", snippet=snippet, kind="DATA", provenance=sunset)
    second = reg.register("https://data.nyc.gov/food", snippet=snippet, kind="DATA", provenance=queens)

    assert first != second
    assert reg.mapping()[second]["provenance"]["snapshot"]["origin_label"].startswith("82nd")


def test_same_url_different_kind_is_distinct():
    reg = CitationRegistry()
    a = reg.register("https://a.gov", snippet="x", kind="DATA")
    b = reg.register("https://a.gov", snippet="x", kind="WEB")
    assert a != b


def test_same_web_evidence_with_different_grade_is_distinct():
    reg = CitationRegistry()
    discovery = reg.register(
        "https://a.gov",
        snippet="same",
        kind="WEB",
        provenance={"evidence_grade": "discovery"},
    )
    authoritative = reg.register(
        "https://a.gov",
        snippet="same",
        kind="WEB",
        provenance={"evidence_grade": "authoritative"},
    )

    assert discovery != authoritative
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


def test_highlight_url_adds_encoded_text_fragment_for_doc_citation():
    citation = Citation(
        id="S1",
        url="https://nyc.gov/program",
        title="Program",
        snippet="The city-run program offers free legal advice today",
        kind="DOC",
    )

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind) == (
        "https://nyc.gov/program#:~:text=The%20city%2Drun%20program%20offers%20free%20legal%20advice%20today"
    )


def test_highlight_url_adds_text_fragment_for_web_citation():
    citation = Citation(
        id="S1",
        url="https://nyc.gov/program",
        title="Program",
        snippet="Official assistance is available through the city today.",
        kind="WEB",
    )

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind).startswith(
        "https://nyc.gov/program#:~:text="
    )


def test_highlight_url_keeps_base_url_for_empty_snippet():
    citation = Citation("S1", "https://nyc.gov/program", "Program", "", "DOC")

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind) == "https://nyc.gov/program"


def test_highlight_url_keeps_base_url_for_whitespace_only_snippet():
    citation = Citation("S1", "https://nyc.gov/program", "Program", " \n\t ", "DOC")

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind) == "https://nyc.gov/program"


def test_highlight_url_keeps_data_citation_url_unchanged():
    citation = Citation("S1", "https://data.cityofnewyork.us/row/1", "Row", "A matching row.", "DATA")

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind) == "https://data.cityofnewyork.us/row/1"


def test_highlight_url_keeps_existing_fragment_unchanged():
    citation = Citation("S1", "https://nyc.gov/program#details", "Program", "Helpful text.", "WEB")

    assert citations.text_fragment_url(citation.url, citation.snippet, citation.kind) == "https://nyc.gov/program#details"


def test_discard_removes_citations_without_reusing_ids():
    """F057: orphaned registrations can be pruned, and a discarded id is never reissued,
    so markers already emitted into text can never silently point at a different source."""
    from heynyc.core.citations import CitationRegistry

    registry = CitationRegistry()
    s1 = registry.register("https://a.example", snippet="kept")
    s2 = registry.register("https://b.example", snippet="orphaned")

    registry.discard({s2})

    assert s1 in registry.mapping()
    assert s2 not in registry.mapping()

    s3 = registry.register("https://c.example", snippet="new after discard")
    assert s3 != s2
    assert registry.mapping()[s3]["url"] == "https://c.example"


def test_legacy_nyc_gov_hosts_normalize_at_registration():
    """F056: the city 301s www1.nyc.gov to www.nyc.gov (verified live 2026-07-19); registering
    the legacy host stores the canonical one so residents and the liveness check never ride a
    deprecated hostname. Only the exact legacy host is rewritten, never path or query."""
    registry = CitationRegistry()
    cid = registry.register("https://www1.nyc.gov/site/hra/help/snap.page?x=1")
    assert registry.mapping()[cid]["url"] == "https://www.nyc.gov/site/hra/help/snap.page?x=1"
    other = registry.register("https://home4.nyc.gov/site/hpd/x.page")
    assert registry.mapping()[other]["url"] == "https://home4.nyc.gov/site/hpd/x.page"


def test_legacy_nyc_gov_hosts_normalize_when_state_is_restored():
    registry = CitationRegistry.from_state({
        "citations": {
            "S1": {
                "id": "S1",
                "url": "https://www1.nyc.gov/site/hra/help/snap.page",
                "title": "SNAP",
                "snippet": "Current guidance",
                "kind": "WEB",
                "valid_as_of": "",
                "provenance": {},
            },
        },
        "counter": 1,
    })

    cite_id = registry.register(
        "https://www.nyc.gov/site/hra/help/snap.page",
        title="SNAP",
        snippet="Current guidance",
        kind="WEB",
    )

    assert cite_id == "S1"
    assert registry.mapping()["S1"]["url"] == (
        "https://www.nyc.gov/site/hra/help/snap.page"
    )
