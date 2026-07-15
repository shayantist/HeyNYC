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


def test_shorter_money_amount_is_not_a_query_restatement():
    """`$800` is not grounded by the user's `$1,800`; digit substrings can change eligibility."""
    snap = {"name": "Supplemental Nutrition Assistance Program"}

    res = check_grounding(
        "Con un ingreso de $800 al mes, revisa SNAP {cite:S1}.",
        {"S1": _data_cite(snap)},
        query="Mi familia gana $1,800 al mes.",
    )

    assert res is not None and res.blocking
    assert res.hard_failures[0].kind == "money"
    assert res.hard_failures[0].text == "$800"


def test_equivalent_money_format_is_grounded():
    snap = {"monthly_income": "$800.00"}

    res = check_grounding(
        "The monthly income is $800 {cite:S1}.",
        {"S1": _data_cite(snap)},
    )

    assert res is not None and res.passed, res.detail


def test_nli_none_is_byte_identical_to_tier1_only():
    """Regression guard for the Tier-2 hook: passing nli=None (today's every caller) must leave the
    Tier-1 result untouched and the two new fields at their empty defaults. Pinned across a grounded
    fixture and a hard-failing one."""
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}

    grounded = check_grounding("Call (917) 720-9700 {cite:S1}.", {"S1": _data_cite(snap)}, nli=None)
    assert grounded is not None
    assert grounded.passed and grounded.blocking is False and not grounded.hard_failures
    assert grounded.nli_failures == [] and grounded.nli_checked == 0

    bad = check_grounding("Call them at (212) 555-0100 {cite:S1}.", {"S1": _data_cite(snap)}, nli=None)
    assert bad is not None and bad.blocking is True and bad.hard_failures
    assert bad.nli_failures == [] and bad.nli_checked == 0

    # Omitting the kwarg entirely is identical to nli=None across every pre-existing field.
    omitted = check_grounding("Call (917) 720-9700 {cite:S1}.", {"S1": _data_cite(snap)})
    assert (omitted.passed, omitted.detail, omitted.blocking, omitted.checked,
            omitted.locations, omitted.hard_failures, omitted.soft_failures) == (
            grounded.passed, grounded.detail, grounded.blocking, grounded.checked,
            grounded.locations, grounded.hard_failures, grounded.soft_failures)
