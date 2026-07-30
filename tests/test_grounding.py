"""Unit tests for the shared grounding module (heynyc.core.grounding).

This is the deterministic cited-claim grounding logic (formerly Part C of eval/checks.py), lifted to a
shared module so BOTH the eval gate and the runtime guard use one implementation. The exhaustive
never-false-fail suite lives in test_checks.py (which drives the same logic through the eval wrapper);
here we assert the PRIMITIVE api the runtime guard consumes: a GroundingResult with structured
hard_failures the guard can turn into feedback / strip on abstain.
"""
from __future__ import annotations

from datetime import date

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


def test_trailing_citation_marker_checks_the_preceding_claim():
    snap = {"hotline": "800-354-0365"}

    res = check_grounding(
        "Call the hotline at 800-354-0365. The service is free.\n\n{cite:S1}",
        {"S1": _data_cite(snap)},
    )

    assert res is not None
    assert res.checked == 1
    assert res.passed


def test_trailing_citation_keeps_failure_at_the_smallest_claim():
    snap = {"hotline": "800-354-0365"}

    res = check_grounding(
        "- Call 800-354-0365.\n- Do not call 212-555-0100.\n\n{cite:S1}",
        {"S1": _data_cite(snap)},
    )

    assert res is not None
    assert res.blocking
    assert res.hard_failures[0].claim == "- Do not call 212-555-0100."


def test_translated_full_date_matches_english_source_by_numeric_components():
    citation = _data_cite(
        {},
        snippet="The termination is effective December 31, 2026.",
    )

    for answer in (
        "Deziyasyon an fini 31 desanm 2026. {cite:S1}",
        "ينتهي التعيين في 31 ديسمبر 2026. {cite:S1}",
    ):
        res = check_grounding(answer, {"S1": citation})
        assert res is not None
        assert res.blocking is False
        assert res.passed


# A date written in Bengali or Arabic-Indic numerals was extracted (`\d` is Unicode-aware) and
# then failed to parse, so it could never match its source and the claim failed grounding in
# exactly the languages rule 11 calls a first-class safety surface.
def test_localized_numeral_date_matches_its_ascii_source():
    citation = _data_cite(
        {},
        snippet="The public charge rule changed on September 25, 2024.",
    )

    for answer in (
        "নিয়মটি ২০২৪-০৯-২৫ তারিখে পরিবর্তিত হয়েছে। {cite:S1}",
        "تبدل القانون في ٢٠٢٤-٠٩-٢٥. {cite:S1}",
    ):
        res = check_grounding(answer, {"S1": citation})
        assert res is not None
        assert res.passed


def test_localized_numeral_date_still_fails_on_a_real_mismatch():
    """Inverse: folding digits must not make a wrong date pass."""
    citation = _data_cite(
        {},
        snippet="The public charge rule changed on September 25, 2024.",
    )

    res = check_grounding("নিয়মটি ২০২৪-০৯-১১ তারিখে পরিবর্তিত হয়েছে। {cite:S1}", {"S1": citation})

    assert res is not None
    assert not res.passed


def test_unparsed_translated_full_date_mismatch_is_soft():
    citation = _data_cite(
        {},
        snippet="The termination is effective July 27, 2026.",
    )

    res = check_grounding(
        "Deziyasyon an fini 3 fevriye 2026.\n\n{cite:S1}",
        {"S1": citation},
    )

    assert res is not None
    assert not res.passed
    assert not res.blocking
    assert res.soft_failures[0].kind == "date"


def test_full_date_mismatch_cannot_hide_behind_same_day_and_year():
    citation = _data_cite(
        {},
        snippet="The termination is effective July 27, 2026.",
    )

    res = check_grounding(
        "The termination is effective February 27, 2026. {cite:S1}",
        {"S1": citation},
    )

    assert res is not None
    assert res.blocking
    assert res.hard_failures[0].kind == "date"


def test_current_date_does_not_need_source_support():
    citation = _data_cite({}, snippet="This page was released July 1, 2026.")

    res = check_grounding(
        "The page cannot confirm whether your status is valid today, July 27, 2026. {cite:S1}",
        {"S1": citation},
        current_date=date(2026, 7, 27),
    )

    assert res is not None
    assert res.passed
    assert any(item["where"] == "system-date" for item in res.locations)


def test_full_month_date_matches_abbreviated_source_month():
    citations = {
        "S1": _data_cite(
            {},
            snippet="The termination notice was published Nov. 28, 2025.",
        )
    }

    supported = check_grounding(
        "The notice was published November 28, 2025. {cite:S1}",
        citations,
    )
    wrong = check_grounding(
        "The notice was published November 29, 2025. {cite:S1}",
        citations,
    )

    assert supported is not None and supported.passed
    assert wrong is not None and wrong.blocking


def test_unparsed_non_english_wrong_month_does_not_pass_as_current_date():
    citation = _data_cite({}, snippet="This page was released July 1, 2026.")

    res = check_grounding(
        "Jodi a se 27 janvye 2026. {cite:S1}",
        {"S1": citation},
        current_date=date(2026, 7, 27),
    )

    assert res is not None
    assert not res.passed
    assert not res.blocking
    assert res.soft_failures[0].kind == "date"
    assert all(item["where"] != "system-date" for item in res.locations)


def test_civic_identifier_that_looks_like_a_date_does_not_block():
    citation = _data_cite({}, snippet="Executive Order 13 protects access to city services.")

    res = check_grounding(
        "Executive Order 13 2026 remains relevant. {cite:S1}",
        {"S1": citation},
    )

    assert res is not None
    assert not res.blocking


def test_exact_resident_money_and_date_may_be_repeated():
    citation = _data_cite({}, snippet="Bring the notice and proof of rent to the appointment.")

    res = check_grounding(
        "Bring your notice dated 3/15/2026 and your $2,400 rent record. {cite:S1}",
        {"S1": citation},
        query="My notice is dated 3/15/2026 and my rent is $2,400.",
    )

    assert res is not None
    assert res.passed
    assert {item["kind"] for item in res.locations if item["where"] == "user-query"} == {
        "date",
        "money",
    }


def test_measured_unit_requires_the_same_unit_category():
    percent = check_grounding(
        "The threshold is 68%. {cite:S1}",
        {"S1": _data_cite({}, snippet="The threshold is 68 degrees.")},
    )
    temperature = check_grounding(
        "The threshold is 68°F. {cite:S1}",
        {"S1": _data_cite({}, snippet="The threshold is 68 percent.")},
    )

    assert percent is not None and percent.blocking
    assert temperature is not None and temperature.blocking


def test_equivalent_measured_unit_wording_passes():
    res = check_grounding(
        "The threshold is 68°F. {cite:S1}",
        {"S1": _data_cite({}, snippet="The threshold is 68 degrees.")},
    )

    assert res is not None and res.passed


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
