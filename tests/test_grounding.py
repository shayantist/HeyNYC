"""Unit tests for the shared grounding module (heynyc.core.grounding).

This is the deterministic cited-claim grounding logic (formerly Part C of eval/checks.py), lifted to a
shared module so BOTH the eval gate and the runtime guard use one implementation. The exhaustive
never-false-fail suite lives in test_checks.py (which drives the same logic through the eval wrapper);
here we assert the PRIMITIVE api the runtime guard consumes: a GroundingResult with structured
hard_failures the guard can turn into feedback / strip on abstain.
"""
from __future__ import annotations

from heynyc.core.citations import content_hash
from heynyc.core.grounding import check_grounding


def _data_cite(snapshot: dict, *, snippet="", url="https://data.cityofnewyork.us/x/row-9.json"):
    return {
        "url": url, "kind": "DATA", "snippet": snippet, "title": "NYC Open Data",
        "provenance": {"record_id": "row-9", "field_pointer": "/",
                       "content_hash": content_hash(snapshot), "snapshot": snapshot},
    }


def test_no_citation_markers_returns_none():
    assert check_grounding("The nearest center is close by.", {}) is None


def test_grounded_answer_passes_and_is_not_blocking():
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    res = check_grounding("Call (917) 720-9700 {cite:S1}.", {"S1": _data_cite(snap)})
    assert res is not None
    assert res.passed and res.blocking is False
    assert not res.hard_failures


def test_fabricated_structured_fact_is_a_blocking_hard_failure():
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    res = check_grounding("Call them at (212) 555-0100 {cite:S1}.", {"S1": _data_cite(snap)})
    assert res is not None
    assert not res.passed and res.blocking is True
    assert res.hard_failures, "the ungrounded phone must be reported as a hard failure"
    fail = res.hard_failures[0]
    # A hard failure carries enough to (a) tell the model exactly what's wrong and (b) strip it.
    assert fail.kind == "phone"
    assert "(212) 555-0100" in fail.text
    assert "S1" in fail.message
    assert fail.claim and fail.claim in "Call them at (212) 555-0100 {cite:S1}."


def test_proper_noun_mismatch_is_soft_not_blocking():
    snap = {"name": "New York Common Pantry", "borough": "Manhattan"}
    res = check_grounding("Try the Bellwether Zephyr Foundation {cite:S1}.", {"S1": _data_cite(snap)})
    assert res is not None
    assert not res.passed          # recorded
    assert res.blocking is False   # a name mismatch never blocks (drift), so the guard won't fire
    assert not res.hard_failures


def test_query_restatement_is_grounded():
    # A fact the user gave us (their origin address) is not a hallucination even if the cited row omits it.
    snap = {"name": "Xavier Mission", "address": "55 West 15th Street"}
    res = check_grounding(
        "The closest pantry to 2920 Broadway is Xavier Mission at 55 West 15th Street {cite:S1}.",
        {"S1": _data_cite(snap)},
        query="Where's the closest food pantry to 2920 Broadway, Manhattan?",
    )
    assert res is not None and res.passed, res.detail
