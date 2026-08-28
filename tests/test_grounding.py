"""Unit tests for the shared grounding module (heynyc.core.grounding).

This is the deterministic cited-claim grounding logic (formerly Part C of eval/checks.py), lifted to a
shared module so BOTH the eval gate and the runtime guard use one implementation. The exhaustive
never-false-fail suite lives in test_checks.py (which drives the same logic through the eval wrapper);
here we assert the PRIMITIVE api the runtime guard consumes: a GroundingResult with structured
hard_failures the guard can turn into feedback / strip on abstain.
"""
from __future__ import annotations

from datetime import date

import pytest

from heynyc.core.citations import CitationRegistry, content_hash
from heynyc.core.grounding import _split_claims, check_grounding, citation_evidence


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


def test_location_citation_evidence_includes_resolved_origin() -> None:
    citation = _data_cite({"address": "575 Fifth Avenue"})
    citation["provenance"]["derivation"] = {
        "origin_query": "Rockefeller Center",
        "origin_label": "Rockefeller Center, 45, Rockefeller Plaza",
    }

    evidence = citation_evidence(citation)

    assert evidence is not None
    assert "Rockefeller Center" in evidence
    assert "45, Rockefeller Plaza" in evidence


def test_web_json_ld_is_source_evidence() -> None:
    evidence = citation_evidence({
        "url": "https://venue.example/jazz-night",
        "kind": "WEB",
        "snippet": "Jazz night",
        "title": "Jazz night",
        "provenance": {
            "structured_data": [{
                "@type": "Event",
                "name": "Jazz night",
                "startDate": "2099-08-28T20:00:00-04:00",
            }],
        },
    })

    assert "2099-08-28T20:00:00-04:00" in evidence


def test_clock_claim_is_not_semantically_parsed():
    snapshot = {"estimated_return": "08/31/2026 11:59:00 PM"}

    result = check_grounding(
        "Estimated return is 11:59 PM. {cite:S1}",
        {"S1": _data_cite(snapshot)},
    )

    assert result is None


def test_url_path_does_not_create_a_structured_date_claim():
    result = check_grounding(
        "See [details](https://www.nycgovparks.org/events/2026/08/14/40-in-focus). {cite:S1}",
        {"S1": _data_cite({}, snippet="An official event details page")},
    )

    assert result is None


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
    assert fail.claim == "Call them at (212) 555-0100."


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


def test_sentence_splitter_keeps_time_abbreviation_with_following_action():
    for text in (
        "The hotline is open until 5 p.m. or call 311 for help.",
        "The hotline is open until 5 P.M. or call 311 for help.",
    ):
        assert _split_claims(text) == [text]


def test_sentence_splitter_still_splits_normal_sentences():
    assert _split_claims("Call 311. Ask for the Tenant Helpline.") == [
        "Call 311.",
        "Ask for the Tenant Helpline.",
    ]


def test_sentence_splitter_keeps_compass_abbreviation_inside_an_address():
    text = "Greeley Square Plaza, Broadway between W. 32nd Street and W. 33rd Street."

    assert _split_claims(text) == [text]


def test_numbered_street_suffix_cannot_change_from_the_source():
    citation = _data_cite({"address": "Broadway between W. 32rd Street and W. 33rd Street"})

    result = check_grounding(
        "Broadway between W. 32nd Street and W. 33rd Street. {cite:S1}",
        {"S1": citation},
    )

    assert result is not None
    assert result.blocking
    assert result.hard_failures[0].kind == "street_ordinal"


def test_coordinated_numbered_street_suffix_cannot_change_from_the_source():
    citation = _data_cite({"address": "Broadway between W. 32rd Street and W. 33rd Street"})

    result = check_grounding(
        "Broadway between W. 32nd and W. 33rd Streets. {cite:S1}",
        {"S1": citation},
    )

    assert result is not None
    assert result.blocking
    assert result.hard_failures[0].kind == "street_ordinal"


def test_markdown_source_link_keeps_the_trailing_citation_on_the_previous_claim():
    citations = {
        "S1": _data_cite({"address": "Broadway between W. 32rd Street and W. 33rd Street"}),
        "S2": _data_cite({"address": "Broadway between W. 34th Street and W. 35th Street"}),
    }

    result = check_grounding(
        "- Broadway between W. 32nd and W. 33rd Streets. "
        "[City listing](https://data.example/row-1) {cite:S1}\n"
        "- Broadway between W. 34th and W. 35th Streets. "
        "[City listing](https://data.example/row-2) {cite:S2}",
        citations,
    )

    assert result is not None
    assert result.blocking
    assert result.hard_failures[0].kind == "street_ordinal"


def test_markdown_source_link_keeps_a_matching_previous_claim_grounded():
    citations = {
        "S1": _data_cite({"address": "Broadway between W. 32nd Street and W. 33rd Street"}),
        "S2": _data_cite({"address": "Broadway between W. 34th Street and W. 35th Street"}),
    }

    result = check_grounding(
        "- Broadway between W. 32nd and W. 33rd Streets. "
        "[City listing](https://data.example/row-1) {cite:S1}\n"
        "- Broadway between W. 34th and W. 35th Streets. "
        "[City listing](https://data.example/row-2) {cite:S2}",
        citations,
    )

    assert result is not None
    assert not result.blocking


def test_street_ordinal_matches_compact_range_and_plural_abbreviation():
    result = check_grounding(
        "The plaza is between 42nd and 43rd Streets. {cite:S1}",
        {
            "S1": _data_cite(
                {},
                snippet="Broadway plaza between 42nd–43rd Sts",
            )
        },
    )

    assert result is not None
    assert result.passed


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
# exactly the languages rule 11 calls a first-class safety surface
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


def test_current_date_still_needs_support_when_claimed_as_an_event_date():
    citation = _data_cite({}, snippet="Movie nights run every Monday.")

    res = check_grounding(
        "Movie night is Wednesday, August 26, 2026. {cite:S1}",
        {"S1": citation},
        current_date=date(2026, 8, 26),
    )

    assert res is not None
    assert res.blocking
    assert res.hard_failures[0].kind == "date"


def test_date_derived_from_resident_weekday_does_not_need_source_support():
    citation = _data_cite({}, snippet="Use the MTA trip planner for an accessible route.")

    for query in (
        "Which accessible route should we take this Saturday?",
        "Is it open on Saturday?",
    ):
        res = check_grounding(
            "For Saturday, August 15, 2026, check the MTA trip planner. {cite:S1}",
            {"S1": citation},
            query=query,
            current_date=date(2026, 8, 13),
        )

        assert res is not None
        assert res.passed
        assert any(item["where"] == "system-date" for item in res.locations)


def test_unrelated_future_date_still_requires_source_support():
    citation = _data_cite({}, snippet="Use the MTA trip planner for an accessible route.")

    res = check_grounding(
        "For Saturday, August 22, 2026, check the MTA trip planner. {cite:S1}",
        {"S1": citation},
        query="Which accessible route should we take this Saturday?",
        current_date=date(2026, 8, 13),
    )

    assert res is not None
    assert res.blocking


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


def test_compact_date_range_requires_support_from_its_own_citation():
    supported = check_grounding(
        "The festival runs August 28–30. {cite:S1}",
        {"S1": _data_cite({}, snippet="The festival returns August 28–30.")},
    )
    wrong_source = check_grounding(
        "The festival runs August 28–30. {cite:S1}",
        {"S1": _data_cite({}, snippet="The festival is a three-day celebration of jazz.")},
    )

    assert supported is not None and supported.passed
    assert wrong_source is not None and wrong_source.blocking
    assert wrong_source.hard_failures[0].kind == "date_range"


def test_compact_date_range_matches_iso_dates_in_structured_evidence():
    result = check_grounding(
        "The festival runs August 28 through 30, 2026. {cite:S1}",
        {
            "S1": _data_cite({
                "start_date": "2026-08-28",
                "end_date": "2026-08-30",
            })
        },
    )

    assert result is not None and result.passed


def test_list_item_citation_checks_every_sentence_on_that_line():
    result = check_grounding(
        (
            "- Charlie Parker Jazz Festival 2026, August 28–30. "
            "It is a three-day jazz festival. {cite:S1}"
        ),
        {"S1": _data_cite({}, snippet="A three-day celebration of live jazz.")},
    )

    assert result is not None and result.blocking
    assert result.hard_failures[0].kind == "date_range"


def test_trailing_list_item_citation_does_not_cover_an_earlier_item():
    result = check_grounding(
        (
            "- First event runs August 28–30\n"
            "- Second event runs September 1–2 {cite:S1}"
        ),
        {"S1": _data_cite({}, snippet="Second event: September 1–2")},
    )

    assert result is not None and result.passed
    assert result.checked == 1


def test_compact_date_range_matches_common_month_abbreviation():
    for claim, source in (
        ("September 28–30, 2026", "Sept 28–30, 2026"),
        ("Sept 28–30, 2026", "September 28–30, 2026"),
        ("Sep 28–30, 2026", "September 28–30, 2026"),
    ):
        result = check_grounding(
            f"The festival runs {claim}. {{cite:S1}}",
            {"S1": _data_cite({}, snippet=f"The festival runs {source}.")},
        )

        assert result is not None and result.passed


@pytest.mark.parametrize(
    ("claim", "source", "wrong_source"),
    [
        (
            "El festival será agosto 28–30, 2026.",
            "Festival: agosto 28–30, 2026.",
            "Festival: agosto 27–30, 2026.",
        ),
        (
            "المهرجان ٢٨–٣٠ أغسطس ٢٠٢٦.",
            "المهرجان ٢٨–٣٠ أغسطس ٢٠٢٦.",
            "المهرجان ٢٧–٣٠ أغسطس ٢٠٢٦.",
        ),
    ],
)
def test_compact_date_range_matches_localized_month_order_and_digits(
    claim, source, wrong_source
):
    supported = check_grounding(
        f"{claim} {{cite:S1}}",
        {"S1": _data_cite({}, snippet=source)},
    )
    mismatched = check_grounding(
        f"{claim} {{cite:S1}}",
        {"S1": _data_cite({}, snippet=wrong_source)},
    )

    assert supported is not None and supported.passed
    assert mismatched is not None and mismatched.blocking
    assert mismatched.hard_failures[0].kind == "date_range"


def test_day_first_compact_date_range_checks_the_month():
    result = check_grounding(
        "المهرجان ٢٨–٣٠ أغسطس ٢٠٢٦. {cite:S1}",
        {"S1": _data_cite({}, snippet="المهرجان ٢٨–٣٠ سبتمبر ٢٠٢٦.")},
    )

    assert result is not None and result.blocking
    assert any(failure.kind == "date_range" for failure in result.hard_failures)


def test_day_first_compact_date_range_allows_a_short_connector():
    supported = check_grounding(
        "El festival será del 28–30 de agosto de 2026. {cite:S1}",
        {"S1": _data_cite({}, snippet="Festival del 28–30 de agosto de 2026.")},
    )
    mismatched = check_grounding(
        "El festival será del 28–30 de agosto de 2026. {cite:S1}",
        {"S1": _data_cite({}, snippet="Festival del 28–30 de septiembre de 2026.")},
    )

    assert supported is not None and supported.passed
    assert mismatched is not None and mismatched.blocking
    assert any(failure.kind == "date_range" for failure in mismatched.hard_failures)


def test_day_first_range_does_not_extract_the_preceding_verb_as_a_month():
    result = check_grounding(
        "The festival runs 28–30 August 2026. {cite:S1}",
        {"S1": _data_cite({}, snippet="The festival takes place 28–30 August 2026.")},
    )

    assert result is not None and result.passed
    assert all(location["token"] != "runs 28–30" for location in result.locations)


def test_day_first_range_without_a_year_checks_the_month():
    supported = check_grounding(
        "The festival runs 28–30 August. {cite:S1}",
        {"S1": _data_cite({}, snippet="The festival takes place 28–30 August.")},
    )
    mismatched = check_grounding(
        "The festival runs 28–30 August. {cite:S1}",
        {"S1": _data_cite({}, snippet="The festival takes place 28–30 September.")},
    )

    assert supported is not None and supported.passed
    assert mismatched is not None and mismatched.blocking
    assert any(failure.kind == "date_range" for failure in mismatched.hard_failures)


@pytest.mark.parametrize(
    "claim",
    [
        "El festival será agosto 28–30, 2026.",
        "المهرجان ٢٨–٣٠ أغسطس ٢٠٢٦.",
    ],
)
def test_localized_date_range_matches_iso_structured_evidence(claim):
    result = check_grounding(
        f"{claim} {{cite:S1}}",
        {
            "S1": _data_cite({
                "start_date": "2026-08-28",
                "end_date": "2026-08-30",
            })
        },
    )

    assert result is not None and result.passed


def test_full_date_matches_yearless_schedule_row_with_same_year_context():
    citation = _data_cite(
        {},
        snippet="2026 schedule: Tue. Oct 20, 7:00 PM EDT, 76ers at Knicks.",
    )

    supported = check_grounding(
        "The next game is October 20, 2026. {cite:S1}",
        {"S1": citation},
    )
    wrong_year = check_grounding(
        "The next game is October 20, 2027. {cite:S1}",
        {"S1": citation},
    )

    assert supported is not None and supported.passed
    assert wrong_year is not None and wrong_year.blocking


def test_human_date_matches_citation_iso_timestamp():
    citation = _data_cite({"facility_name": "Sobelsohn Park"})
    citation["valid_as_of"] = "2025-06-27T13:37:17.684Z"

    supported = check_grounding(
        "The record is dated June 27, 2025. {cite:S1}",
        {"S1": citation},
    )
    wrong = check_grounding(
        "The record is dated June 28, 2025. {cite:S1}",
        {"S1": citation},
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


def test_resident_weekday_is_not_semantically_parsed():
    citation = _data_cite({}, snippet="A landlord needs a court order to evict a tenant.")

    res = check_grounding(
        "A demand to leave by Friday is not a court order. {cite:S1}",
        {"S1": citation},
        query="My landlord says I have to leave by Friday.",
    )

    assert res is None


def test_current_weekday_is_not_semantically_parsed():
    citation = _data_cite({}, snippet="A landlord needs a court order to evict a tenant.")

    res = check_grounding(
        "Tuesday is the deadline. {cite:S1}",
        {"S1": citation},
        query="My landlord says I have to leave by Friday.",
        current_date=date(2026, 8, 11),
    )

    assert res is None


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


def test_schema_org_usd_price_is_grounded_from_search_evidence():
    citation = {
        "url": "https://example.com/events",
        "kind": "WEB",
        "title": "Concert listings",
        "snippet": (
            '{"name":"Candlelight: 90s Hip Hop on Strings",'
            '"offers":{"lowPrice":28,"priceCurrency":"USD"}}'
        ),
        "provenance": {"evidence_grade": "search_excerpt"},
    }

    res = check_grounding(
        "Tickets are listed from $28. {cite:S1}",
        {"S1": citation},
    )
    citation["snippet"] = citation["snippet"].replace('"USD"', '"EUR"')
    wrong_currency = check_grounding(
        "Tickets are listed from $28. {cite:S1}",
        {"S1": citation},
    )

    assert res is not None and res.passed, res.detail
    assert wrong_currency is not None and wrong_currency.blocking


def test_f177_map_coordinates_must_match_one_cited_record():
    citations = {
        "S1": _data_cite({"name": "Hunts Point", "latitude": "40.817656", "longitude": "-73.890358"}),
        "S2": _data_cite({"name": "East End", "latitude": "40.804465", "longitude": "-73.93526"}),
    }

    res = check_grounding(
        "[Mapa de East End](https://www.google.com/maps/search/?api=1&query=40.80447,-73.89036) "
        "{cite:S1} {cite:S2}",
        citations,
    )

    assert res is not None and res.blocking
    assert res.hard_failures[0].kind == "map_coordinates"


def test_f177_map_coordinates_accept_rounded_cited_record():
    citation = _data_cite({"name": "East End", "latitude": "40.804465", "longitude": "-73.93526"})

    res = check_grounding(
        "[Mapa de East End](https://www.google.com/maps/search/?api=1&query=40.80447,-73.93526) "
        "{cite:S1}",
        {"S1": citation},
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


def test_citation_can_be_persisted_without_exposing_it_in_the_current_turn():
    citations = CitationRegistry()
    cursor = citations.touch_cursor()

    citation_id = citations.register(
        "https://events.example/later",
        title="Later event",
        snippet="A saved candidate for a later turn.",
        touch=False,
    )

    assert citation_id in citations.mapping()
    assert citations.touched_since(cursor) == set()
    assert citations.touch(citation_id)
    assert citations.touched_since(cursor) == {citation_id}
