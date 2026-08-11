"""Offline tests for check_cited_claim_grounding (Part C).

The load-bearing property under test is the #1 rule: this check must NOT false-fail a grounded
answer. So the bulk of these cases assert that correctly-grounded claims (in a DATA snapshot, a
DOC/WEB snippet, or restated from the user's own query) PASS, across every normalization the check
promises (phone formats, "68°F" vs "68 degrees", punctuated proper nouns). A smaller set asserts
that a genuinely fabricated phone / address / number / name still FAILS.
"""
from __future__ import annotations

from types import SimpleNamespace

from heynyc.core.citations import content_hash
from heynyc.eval.cases import EvalCase
from heynyc.eval.checks import (
    _CITED_CLAIM_GROUNDING_BLOCKING,
    check_cited_claim_grounding,
    check_link_liveness,
    check_turn_completion,
    run_checks,
)
from heynyc.eval.runner import CaseResult


def _case(query: str = "") -> EvalCase:
    return EvalCase(id="c", module="m", query=query)


def _result(text, citations=None, query="") -> CaseResult:
    return CaseResult(case=_case(query), text=text, citations=citations or {})


def _data_cite(snapshot: dict, *, snippet="", title="NYC Open Data (h2bn-gu9k)", url="https://data.cityofnewyork.us/x/row-9.json"):
    return {
        "url": url, "kind": "DATA", "snippet": snippet, "title": title,
        "provenance": {"record_id": "row-9", "field_pointer": "/",
                       "content_hash": content_hash(snapshot), "snapshot": snapshot},
    }


def _doc_cite(snippet: str, *, title="How to apply for SNAP", kind="DOC", url="https://www.nyc.gov/snap"):
    return {"url": url, "kind": kind, "snippet": snippet, "title": title}


# --- no-op / skip conditions ----------------------------------------------

def test_incomplete_approval_turn_blocks_the_eval_gate():
    result = _result("")
    result.turn_results = [
        SimpleNamespace(text="First answer", status="success"),
        SimpleNamespace(text="", status="approval_required"),
    ]

    check = check_turn_completion(result)

    assert check is not None and not check.passed
    assert check.blocking
    assert "turn 2" in check.detail
    assert "approval_required" in check.detail


def test_turn_completion_requires_explicit_success_and_accepts_normal_turns():
    missing_status = _result("Answer")
    missing_status.turn_results = [SimpleNamespace(text="Answer")]
    successful = _result("Answer")
    successful.turn_results = [
        SimpleNamespace(text="First answer", status="success"),
        SimpleNamespace(text="Second answer", status="success"),
    ]

    assert not check_turn_completion(missing_status).passed
    assert check_turn_completion(successful).passed


def test_no_citation_markers_returns_none():
    assert check_cited_claim_grounding(_result("The nearest center is close by.")) is None


async def test_link_liveness_strips_internal_url_marker_brace():
    checked = []

    async def checker(url: str) -> int:
        checked.append(url)
        return 200

    result = await check_link_liveness(
        _result("Official page {URL:https://www.nyc.gov/help}"),
        checker,
    )

    assert result is not None and result.passed
    assert checked == ["https://www.nyc.gov/help"]


def test_no_salient_tokens_returns_none():
    # A cited claim with no high-signal facts (just a lowercase generic) → nothing to verify.
    cr = _result("Yes, you can get help there {cite:S1}.",
                 citations={"S1": _doc_cite("Homebase helps tenants facing eviction.")})
    assert check_cited_claim_grounding(cr) is None


def test_unverifiable_citation_is_skipped_not_failed():
    # A DATA citation with neither snapshot nor snippet can't be verified → must NOT fail.
    bare = {"S1": {"url": "https://a.gov", "kind": "DATA"}}
    cr = _result("Call them at (718) 557-1379 {cite:S1}.", citations=bare)
    assert check_cited_claim_grounding(cr) is None  # phone token, but source has no content → skip


# --- DATA: grounded PASSES -------------------------------------------------

def test_data_grounded_phone_address_name_pass():
    snap = {"name": "New York Common Pantry", "address": "8 East 109th Street",
            "borough": "Manhattan", "phone": "(917) 720-9700", "status": "Open"}
    cr = _result(
        "The nearest food pantry is New York Common Pantry at 8 East 109th Street {cite:S1}. "
        "You can reach them at (917) 720-9700 {cite:S1}.",
        citations={"S1": _data_cite(snap, snippet="New York Common Pantry — Manhattan (status: Open)")},
    )
    res = check_cited_claim_grounding(cr)
    assert res is not None and res.passed, res.detail
    # location recording: every matched fact carries a where-pointer.
    assert res.locations
    assert any(loc["where"].startswith("S1#") for loc in res.locations)


def test_data_phone_format_normalization_passes():
    # Source stores dashes; answer prints parens+space → same digit-run, must match.
    snap = {"name": "Sedgwick Library", "phone": "718-579-2179"}
    cr = _result("Reach Sedgwick Library at (718) 579-2179 {cite:S1}.",
                 citations={"S1": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail


def test_origin_address_from_query_does_not_false_fail():
    # The user's origin address is restated in the claim but is NOT in the cited pantry's row.
    # It's grounded in the QUERY, so it must pass (restating the user is not a hallucination).
    snap = {"name": "Xavier Mission", "address": "55 West 15th Street", "borough": "Manhattan"}
    cr = _result(
        "The closest pantry to 2920 Broadway is Xavier Mission at 55 West 15th Street {cite:S1}.",
        citations={"S1": _data_cite(snap)},
        query="Where's the closest food pantry to 2920 Broadway, Manhattan?",
    )
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail
    assert any(loc["where"] == "user-query" for loc in res.locations)


def test_neighborhood_proper_noun_from_query_passes():
    snap = {"name": "Part of the Solution", "borough": "Bronx"}
    cr = _result("Near Fordham Road, try Part of the Solution {cite:S1}.",
                 citations={"S1": _data_cite(snap)},
                 query="I'm near Fordham Road in the Bronx and I need free food.")
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail


def test_generic_geographic_proper_noun_not_extracted():
    # "New York City" / borough names are boilerplate, not source-specific facts → never extracted,
    # so an answer that mentions them while citing a minimal row does not false-fail.
    snap = {"name": "Cooling Center", "borough": "Bronx"}
    cr = _result("There is a cooling center in New York City in the Bronx {cite:S1}.",
                 citations={"S1": _data_cite(snap)})
    # Only "Cooling Center" is verifiable and it IS in the snapshot name; NYC/Bronx are skipped.
    res = check_cited_claim_grounding(cr)
    assert res is None or res.passed, (res.detail if res else "")


def test_spanish_generic_geographic_proper_noun_not_extracted():
    cr = _result(
        "En la ciudad de Nueva York, Section 8 está protegido {cite:S1}.",
        citations={"S1": _doc_cite("Section 8 is protected in New York City.", kind="WEB")},
    )

    res = check_cited_claim_grounding(cr)

    assert res is None or res.passed, (res.detail if res else "")

    capitalized = _result(
        "La regla se aplica en la Ciudad de Nueva York {cite:S1}.",
        citations={"S1": _doc_cite("The rule applies in New York City.", kind="WEB")},
    )
    res = check_cited_claim_grounding(capitalized)
    assert res is None or res.passed, (res.detail if res else "")


# --- HARD fabrications FAIL and BLOCK (verbatim structured facts) ----------

def test_fabricated_phone_fails_and_blocks():
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    cr = _result("Call them at (212) 555-0100 {cite:S1}.",  # not the row's number
                 citations={"S1": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert res is not None and not res.passed
    assert "phone" in res.detail and "S1" in res.detail
    assert res.blocking is _CITED_CLAIM_GROUNDING_BLOCKING  # a verbatim fact → blocks the gate


async def test_semantic_string_matcher_is_not_a_release_gate() -> None:
    async def live(_url: str) -> int:
        return 200

    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    cr = _result(
        "Call them at (212) 555-0100 {cite:S1}.",
        citations={"S1": _data_cite(snap)},
    )

    checks = await run_checks(cr, link_checker=live)

    assert "cited_claim_grounding" not in {check.name for check in checks}


def test_fabricated_address_fails_and_blocks():
    snap = {"name": "Xavier Mission", "address": "55 West 15th Street", "borough": "Manhattan"}
    cr = _result("It's at 900 Nonexistent Boulevard {cite:S1}.",
                 citations={"S1": _data_cite(snap)},
                 query="Where is the nearest pantry to Union Square?")
    res = check_cited_claim_grounding(cr)
    assert not res.passed and "address" in res.detail
    assert res.blocking is _CITED_CLAIM_GROUNDING_BLOCKING


def test_fabricated_dollar_amount_fails_and_blocks():
    snap = {"program": "SNAP", "monthly_benefit": "$291"}
    cr = _result("You could get $1,632 a month {cite:S1}.",
                 citations={"S1": _data_cite(snap, title="Benefits")})
    res = check_cited_claim_grounding(cr)
    assert not res.passed and "money" in res.detail
    assert res.blocking is _CITED_CLAIM_GROUNDING_BLOCKING


def test_dollar_amount_does_not_match_an_unrelated_bare_year():
    cr = _result(
        "The benefit is $2,026 {cite:S1}.",
        citations={"S1": _doc_cite("Updated January 2026.", kind="WEB")},
    )

    res = check_cited_claim_grounding(cr)

    assert res is not None and not res.passed and res.blocking


def test_dollar_amount_does_not_match_a_bare_day_in_page_prose():
    cr = _result(
        "The fee is $27 {cite:S1}.",
        citations={"S1": _doc_cite("Updated July 27, 2026.", kind="WEB")},
    )

    res = check_cited_claim_grounding(cr)

    assert res is not None and not res.passed and res.blocking


def test_dollar_amount_matches_a_bare_structured_value():
    cr = _result(
        "The benefit is $291 {cite:S1}.",
        citations={"S1": _data_cite({"monthly_benefit": 291})},
    )

    res = check_cited_claim_grounding(cr)

    assert res is not None and res.passed


def test_phone_repeated_from_query_still_needs_citation_evidence():
    cr = _result(
        "Call (212) 555-0100 {cite:S1}.",
        citations={"S1": _doc_cite("Call the official helpline.", kind="WEB")},
        query="Is (212) 555-0100 the right number?",
    )

    res = check_cited_claim_grounding(cr)

    assert res is not None and not res.passed and res.blocking


# --- DOC / WEB: snippet grounding -----------------------------------------

def test_doc_snippet_grounded_passes():
    snippet = ("Right to Counsel gives tenants a free lawyer in Housing Court. "
               "Call the Office of Civil Justice at 718-557-1379 for help.")
    cr = _result(
        "Under Right to Counsel you can get a free lawyer {cite:S1}. "
        "Call the Office of Civil Justice at (718) 557-1379 {cite:S1}.",
        citations={"S1": _doc_cite(snippet, title="Right to Counsel — NYC")},
    )
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail


def test_web_snippet_fabricated_number_blocks():
    # A citation supports only the evidence captured for it. An exact fact absent from that evidence
    # must not ship on the theory that it might appear elsewhere on the page.
    snippet = "The heat advisory is in effect with a high near 68 degrees today."
    cr = _result("There's a heat advisory with highs near 95°F {cite:S1}.",
                 citations={"S1": _doc_cite(snippet, kind="WEB", title="Notify NYC")})
    res = check_cited_claim_grounding(cr)
    assert not res.passed and "unit_number" in res.detail
    assert res.blocking is _CITED_CLAIM_GROUNDING_BLOCKING


def test_label_only_data_citation_does_not_support_an_exact_phone():
    # A correct phone cited to the wrong evidence is still unsupported. The model must retrieve and
    # cite the source that actually contains the number.
    catalog = {"url": "https://a.gov", "kind": "DATA",
               "snippet": "Supplemental Nutrition Assistance Program — Money to buy food",
               "title": "Supplemental Nutrition Assistance Program", "provenance": {}}
    cr = _result("Apply for SNAP; call the HRA Infoline at 718-557-1399 {cite:S1}.",
                 citations={"S1": catalog})
    res = check_cited_claim_grounding(cr)
    assert res.blocking is _CITED_CLAIM_GROUNDING_BLOCKING


def test_temperature_unit_normalization_passes():
    # "68°F" in the answer vs "68 degrees" in the source → same number, must match.
    snippet = "Highs today near 68 degrees under an air quality advisory."
    cr = _result("It's about 68°F right now {cite:S1}.",
                 citations={"S1": _doc_cite(snippet, kind="WEB", title="Notify NYC")})
    assert check_cited_claim_grounding(cr).passed


def test_proper_noun_with_punctuation_normalizes_and_passes():
    # Source stores "St. Mary's Church"; answer writes it the same way but the check must survive the
    # apostrophe/period punctuation on both sides.
    snap = {"name": "St. Mary's Church Pantry", "borough": "Bronx"}
    cr = _result("Try St. Mary's Church Pantry {cite:S1}.", citations={"S1": _data_cite(snap)})
    assert check_cited_claim_grounding(cr).passed


def test_proper_noun_word_fallback_passes_when_phrase_not_contiguous():
    # The full phrase isn't a contiguous substring of the snapshot (fields split the words), but all
    # significant words are present → the fallback keeps this from false-failing.
    snap = {"org": "Common Pantry", "program": "New York", "extra": "food help"}
    cr = _result("Go to New York Common Pantry {cite:S1}.", citations={"S1": _data_cite(snap)})
    assert check_cited_claim_grounding(cr).passed


def test_agent_corrected_source_typo_does_not_false_fail():
    # Real trace regression (food_ungrounded_hours): the FoodHelp row stores the name with a typo
    # ("...FOOD PANTY"); the agent wrote the correct "...Food Pantry". 4 of 5 significant words still
    # match the same row — a grounded answer that must NOT be flagged (the #1 rule).
    snap = {"FID": 660, "program": "MANHATTAN DOWNTOWN MULTI-CULTURAL FOOD PANTY",
            "org_phone": "(347) 248-9726", "distadd": "160 Allen St, New York, NY, 10002, USA"}
    cr = _result(
        "**Manhattan Downtown Multi-Cultural Food Pantry** — 160 Allen St {cite:S3} | (347) 248-9726",
        citations={"S3": _data_cite(snap, snippet="MANHATTAN DOWNTOWN MULTI-CULTURAL FOOD PANTY — 160 Allen St")},
    )
    res = check_cited_claim_grounding(cr)
    assert res is None or res.passed, (res.detail if res else "")


def test_address_ordinal_suffix_drift_does_not_false_fail():
    # Real trace regression (cooling_nearest_address): source stores "221 W 107 St"; the agent wrote
    # "221 W 107th St". Same building — the ordinal suffix must not false-fail.
    snap = {"Facility_name": "ABSW OAC", "Address": "221 W 107 St", "Borough_name": "Manhattan"}
    cr = _result("The older-adult center is at 221 W 107th St {cite:S2}.",
                 citations={"S2": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail
    assert any(loc["kind"] == "address" and loc["where"] == "S2#/Address" for loc in res.locations)


def test_street_type_abbreviation_drift_does_not_false_fail():
    # Real trace regressions (cooling): source "2070 Clinton Av" vs agent "Clinton Ave", and source
    # "515 Malcolm X Boulevard" vs agent "Malcolm X Blvd". The street name is grounded; the type word
    # (Av/Ave, Boulevard/Blvd) drifts and must never be a required match word.
    snap_a = {"Facility_name": "Thomas Guess OAC", "Address": "2070 Clinton Av", "Borough_name": "Bronx"}
    cr_a = _result("Thomas Guess OAC — 2070 Clinton Ave, 0.81 mi away {cite:S6}",
                   citations={"S6": _data_cite(snap_a)})
    assert check_cited_claim_grounding(cr_a).passed

    snap_b = {"Facility_name": "Schomburg Center for Research in Black Culture",
              "Address": "515 Malcolm X Boulevard", "Borough_name": "Manhattan"}
    cr_b = _result("Schomburg Center for Research in Black Culture — 515 Malcolm X Blvd {cite:S2}",
                   citations={"S2": _data_cite(snap_b)})
    assert check_cited_claim_grounding(cr_b).passed


def test_fabricated_proper_noun_name_fails_but_does_not_block():
    # A wholly fabricated name still fails the check (recorded), but a proper-noun mismatch is SOFT —
    # names drift too much (acronyms, neighborhoods, aggregation) to block the gate on. So passed is
    # False yet blocking is False. The hard structured facts are what block.
    snap = {"name": "New York Common Pantry", "borough": "Manhattan"}
    cr = _result("Try the Bellwether Zephyr Foundation {cite:S1}.",
                 citations={"S1": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert not res.passed and "proper_noun" in res.detail
    assert res.blocking is False


# --- proper-noun drift: correct answers must not be flagged ----------------

def test_date_abbreviation_is_not_a_salient_fact():
    # "Fri Jul 3" is a date the agent formats — never a fact the cited event row must contain.
    snap = {"event": "Carnegie Hall Citywide", "venue": "Bryant Park", "date": "2026-07-03"}
    cr = _result("Carnegie Hall Citywide — Fri Jul 3 at Bryant Park {cite:S1}.",
                 citations={"S1": _data_cite(snap, title="Bryant Park event")})
    res = check_cited_claim_grounding(cr)
    assert res is None or res.passed, (res.detail if res else "")


def test_citation_valid_as_of_supports_the_rendered_source_date():
    cr = _result(
        "The listing is dated 2026-04-11 {cite:S1}.",
        citations={
            "S1": {
                "url": "https://example.gov/row",
                "kind": "DATA",
                "snippet": "City-listed food pantry",
                "valid_as_of": "2026-04-11",
            }
        },
    )

    result = check_cited_claim_grounding(cr)

    assert result is not None and result.passed, result.detail


def test_plural_singular_drift_passes():
    # Answer "Picnic Performances" vs source title "…Picnic Performance" (singular) → grounded.
    cr = _result("Catch the Bryant Park Picnic Performances series {cite:S1}.",
                 citations={"S1": _doc_cite("Bryant Park Picnic Performance: The Knights",
                                            title="Bryant Park Picnic Performance")})
    assert check_cited_claim_grounding(cr).passed


def test_correct_neighborhood_not_in_row_does_not_block():
    # snap_where_to_apply_in_person regression: the agent adds the correct neighborhood "Far Rockaway"
    # (right for that address) though the row only says "Queens". "Rockaway" IS in the facility name,
    # so allow-one-miss passes it; even if not, a name mismatch is soft and never blocks.
    snap = {"facility_name": "Rockaway SNAP Center", "borough": "Queens",
            "street_address": "219 Beach 59th Street, 1st Fl."}
    cr = _result("The Rockaway SNAP Center is at 219 Beach 59th Street in Far Rockaway {cite:S1}.",
                 citations={"S1": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert res.passed or res.blocking is False, res.detail


def test_distance_plus_origin_not_parsed_as_address():
    # snap_nearest_address_manhattan regression: "0.9 miles from 2920 Broadway" must NOT be misparsed
    # as a fake street address "9 miles from 2920 Broadway". The real address IS in the row.
    snap = {"facility_name": "HRA Benefits Access Center St. Nicholas",
            "street_address": "132 W. 125th Street, 4th Fl.", "borough": "Manhattan"}
    cr = _result("132 W. 125th Street, 4th Floor — about 0.9 miles from 2920 Broadway {cite:S1}.",
                 citations={"S1": _data_cite(snap)},
                 query="Closest SNAP center to 2920 Broadway, Manhattan?")
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail


# --- multi-source & conservatism ------------------------------------------

def test_fact_matched_across_union_of_cited_sources():
    # A claim cites two sources; each fact lives in one of them. Neither should be flagged.
    s1 = _data_cite({"name": "Marconi Park", "borough": "Bronx"})
    s2 = _data_cite({"name": "Rodney Park North", "phone": "(718) 555-2222"})
    cr = _result("Cool off at Marconi Park; call Rodney Park North at (718) 555-2222 {cite:S1}{cite:S2}.",
                 citations={"S1": s1, "S2": s2})
    res = check_cited_claim_grounding(cr)
    assert res.passed, res.detail


def test_empty_citation_does_not_rescue_an_unsupported_exact_fact():
    # An empty citation contains no evidence and cannot rescue an exact fact absent from the other
    # cited source.
    s1 = {"url": "https://a.gov", "kind": "DATA"}  # no snapshot/snippet
    s2 = _data_cite({"name": "Marconi Park", "borough": "Bronx"})
    cr = _result("Reach the site at (646) 999-1234 {cite:S1}{cite:S2}.",
                 citations={"S1": s1, "S2": s2})
    res = check_cited_claim_grounding(cr)
    assert res is not None and res.blocking, res.detail


def test_api_provenance_response_is_used_as_source_content():
    # Benefits cites an auditable API exchange (api_provenance): the captured `response` is the source
    # content, so a program name / amount stated from it must verify against `response`, not just the
    # snippet. Distinct from a DATA snapshot but treated the same way.
    api_cite = {
        "url": "https://a.gov", "kind": "DATA", "snippet": "NYC Benefits Screening — likely eligible",
        "title": "NYC Benefits Screening (ACCESS NYC)",
        "provenance": {"endpoint": "POST /eligibilityPrograms",
                       "response": {"eligiblePrograms": [{"name": "Supplemental Nutrition Assistance Program"}]}},
    }
    cr = _result("You're likely eligible for the Supplemental Nutrition Assistance Program {cite:S1}.",
                 citations={"S1": api_cite})
    assert check_cited_claim_grounding(cr).passed


def test_grounded_answer_is_not_blocking():
    # A fully grounded answer passes and (having no hard failure) is not marked blocking.
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    cr = _result("Call (917) 720-9700 {cite:S1}.", citations={"S1": _data_cite(snap)})
    res = check_cited_claim_grounding(cr)
    assert res.passed and res.blocking is False


def test_flesch_kincaid_treats_line_breaks_as_sentence_ends():
    """Phone-format answers are dash lists with no terminal periods (the voice rules demand
    exactly that); the splitter must not score six short lines as one forty-word run-on."""
    from heynyc.eval.checks import flesch_kincaid_grade

    bullets = (
        "Here is what to do next for your food help today\n"
        "- Bring your photo ID and your mail\n"
        "- Go to the office before four today\n"
        "- Ask the desk for a same day slot\n"
        "- Call this number if the line is long\n"
        "- Keep your papers in one safe place\n"
        "- Text me after and I can help more"
    )
    prose = bullets.replace("\n- ", ". ").replace("\n", ". ")

    bullet_grade = flesch_kincaid_grade(bullets)
    prose_grade = flesch_kincaid_grade(prose)

    assert bullet_grade is not None and prose_grade is not None
    assert abs(bullet_grade - prose_grade) < 1.0  # a line break ends a thought
    assert bullet_grade <= 9
