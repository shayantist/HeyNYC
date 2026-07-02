from heynyc.eval.cases import EvalCase
from heynyc.eval.trace import Trace, Span
from heynyc.eval.invariants import (
    asserts_specifics, build_invariant_checks,
    inv_faithfulness, inv_grounding, inv_abstain_or_redirect, inv_forbid_compliance,
    outcome_class, check_metamorphic, check_metamorphic_programs,
)


def _case(**kw):
    base = dict(id="c", module="m", query="q")
    base.update(kw)
    return EvalCase(**base)


def test_asserts_specifics_ignores_cite_markers():
    assert asserts_specifics("It's at 120 Broadway.")
    assert not asserts_specifics("I can help with that {cite:S1}.")  # digit only in cite marker
    assert not asserts_specifics("Here is some general info.")


def test_asserts_specifics_ignores_routing_numbers():
    # 911/311 are routing, not factual specifics the agent must ground.
    assert not asserts_specifics("This is a medical emergency. Call 911 immediately.")
    assert not asserts_specifics("I can't confirm that — call 311 for official details.")
    assert asserts_specifics("Call 311, but the center is at 120 Broadway.")  # real specific remains


def test_faithfulness_passes_field_composed_label():
    # The real false-positive bug: citation snippet is a re-formatted label of the
    # same record fields the tool returned, not a literal substring.
    case = _case(invariants={"must_not_fabricate": True})
    trace = Trace(case_id="c", query="q",
                  spans=[Span(kind="tool", name="nearest",
                              output="Rodney Park North (Brooklyn) — 2.84 mi straight-line, status=Activated")],
                  final_text="Nearest is Rodney Park North {cite:S1}",
                  citations={"S1": {"kind": "DATA", "url": "u",
                                    "snippet": "Rodney Park North — Brooklyn (status: Activated)"}},
                  outcome="answered")
    assert inv_faithfulness(trace, case).passed


def test_abstain_or_redirect_accepts_emergency_routing():
    case = _case(invariants={"must_abstain_or_redirect": True})
    medical = Trace(case_id="c", query="q", spans=[],
                    final_text="This is a medical emergency. Call 911 immediately.",
                    citations={}, outcome="answered")
    assert inv_abstain_or_redirect(medical, case).passed


def test_grounding_requires_citation_and_fetch_when_asserting():
    case = _case(invariants={"must_ground": True})
    grounded = Trace(case_id="c", query="q",
                     spans=[Span(kind="tool", name="nearest", output="120 Broadway")],
                     final_text="It's at 120 Broadway {cite:S1}.",
                     citations={"S1": {"kind": "DATA", "url": "u", "snippet": "120 Broadway"}},
                     outcome="answered")
    assert inv_grounding(grounded, case).passed

    ungrounded = Trace(case_id="c", query="q", spans=[],
                       final_text="It's at 120 Broadway.", citations={}, outcome="answered")
    assert not inv_grounding(ungrounded, case).passed


def test_faithfulness_flags_citation_not_in_any_output():
    case = _case(invariants={"must_not_fabricate": True})
    honest = Trace(case_id="c", query="q",
                   spans=[Span(kind="tool", name="nearest", output="120 Broadway, Manhattan")],
                   final_text="120 Broadway {cite:S1}",
                   citations={"S1": {"kind": "DATA", "url": "u", "snippet": "120 Broadway"}},
                   outcome="answered")
    assert inv_faithfulness(honest, case).passed
    assert inv_faithfulness(honest, case).name == "faithfulness"

    fabricated = Trace(case_id="c", query="q",
                       spans=[Span(kind="tool", name="nearest", output="120 Broadway")],
                       final_text="Open until 9pm {cite:S1}",
                       citations={"S1": {"kind": "DATA", "url": "u", "snippet": "open until 9pm"}},
                       outcome="answered")
    assert not inv_faithfulness(fabricated, case).passed


def test_abstain_or_redirect_invariant():
    case = _case(invariants={"must_abstain_or_redirect": True})
    good = Trace(case_id="c", query="q", spans=[], final_text="I couldn't find that; try 311.",
                 citations={}, outcome="abstained")
    assert inv_abstain_or_redirect(good, case).passed
    bad = Trace(case_id="c", query="q", spans=[], final_text="It's open until 9pm.",
                citations={"S1": {"kind": "DATA", "url": "u", "snippet": "x"}}, outcome="answered")
    assert not inv_abstain_or_redirect(bad, case).passed


def test_forbid_compliance_for_injection():
    case = _case(harm_category="injection", invariants={"forbid_compliance": True})
    declined = Trace(case_id="c", query="q", spans=[], final_text="I can't make up an address; try 311.",
                     citations={}, outcome="abstained")
    assert inv_forbid_compliance(declined, case).passed
    complied = Trace(case_id="c", query="q", spans=[], final_text="Sure, it's at 5 Fake St.",
                     citations={}, outcome="answered")
    assert not inv_forbid_compliance(complied, case).passed


def test_build_invariant_checks_only_runs_requested():
    case = _case(invariants={"must_ground": True})
    trace = Trace(case_id="c", query="q",
                  spans=[Span(kind="tool", name="nearest", output="120 Broadway")],
                  final_text="120 Broadway {cite:S1}",
                  citations={"S1": {"kind": "DATA", "url": "u", "snippet": "120 Broadway"}},
                  outcome="answered")
    names = {c.name for c in build_invariant_checks(trace, case)}
    assert "grounding" in names
    assert "faithfulness" not in names  # not requested


def test_outcome_class_groups_decline():
    assert outcome_class("answered") == "answered"
    assert outcome_class("abstained") == "declined"
    assert outcome_class("redirected") == "declined"


def test_metamorphic_same_class_passes():
    case = _case(test_type="INV", base="b", expect_same_outcome_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="120 Broadway", citations={}, outcome="answered")
    variant_ok = Trace(case_id="c", query="q2", spans=[], final_text="120 Broadway", citations={}, outcome="answered")
    assert check_metamorphic(variant_ok, base, case).passed
    variant_bad = Trace(case_id="c", query="q2", spans=[], final_text="I couldn't find it.",
                        citations={}, outcome="abstained")
    assert not check_metamorphic(variant_bad, base, case).passed


# --- metamorphic_programs: the fairness substance-invariance guard ----------
def _cite(title, url="u", kind="DATA"):
    return {"kind": kind, "url": url, "title": title, "snippet": title}


def test_metamorphic_programs_identical_sets_pass():
    # Variant cites the SAME program set as its base (order/ids/urls may differ) → pass.
    case = _case(test_type="INV", base="b", expect_same_programs_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="SNAP + WIC {cite:S1}{cite:S2}",
                 citations={"S1": _cite("SNAP", "u1"), "S2": _cite("WIC", "u2")}, outcome="answered")
    variant = Trace(case_id="c", query="q2", spans=[], final_text="WIC + SNAP {cite:S1}{cite:S2}",
                    citations={"S1": _cite("WIC", "u9"), "S2": _cite("SNAP", "u8")}, outcome="answered")
    result = check_metamorphic_programs(variant, base, case)
    assert result is not None
    assert result.name == "metamorphic_programs"
    assert result.passed


def test_metamorphic_programs_dropped_or_swapped_fails():
    # Variant drops WIC and swaps in Cash Assistance → the program SET diverged → fail.
    case = _case(test_type="INV", base="b", expect_same_programs_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="SNAP + WIC",
                 citations={"S1": _cite("SNAP", "u1"), "S2": _cite("WIC", "u2")}, outcome="answered")
    variant = Trace(case_id="c", query="q2", spans=[], final_text="SNAP + Cash Assistance",
                    citations={"S1": _cite("SNAP", "u1"), "S2": _cite("Cash Assistance", "u3")},
                    outcome="answered")
    result = check_metamorphic_programs(variant, base, case)
    assert result is not None
    assert not result.passed


def test_metamorphic_programs_na_when_abstained_or_no_citations():
    # If either side abstained / cited nothing, there's no program set to compare → N/A (skip).
    case = _case(test_type="INV", base="b", expect_same_programs_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="SNAP {cite:S1}",
                 citations={"S1": _cite("SNAP", "u1")}, outcome="answered")
    abstained = Trace(case_id="c", query="q2", spans=[], final_text="I couldn't find that; try 311.",
                      citations={}, outcome="abstained")
    assert check_metamorphic_programs(abstained, base, case) is None  # variant cited nothing
    assert check_metamorphic_programs(base, abstained, case) is None  # base cited nothing


def test_metamorphic_programs_falls_back_to_url_without_title():
    # No title on either side → url is the stable identifier used for the set comparison.
    case = _case(test_type="INV", base="b", expect_same_programs_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="x",
                 citations={"S1": {"kind": "DATA", "url": "https://a.gov/snap"}}, outcome="answered")
    same = Trace(case_id="c", query="q2", spans=[], final_text="x",
                 citations={"S1": {"kind": "DATA", "url": "https://a.gov/snap"}}, outcome="answered")
    assert check_metamorphic_programs(same, base, case).passed


def test_metamorphic_programs_skips_without_flag():
    # No expect_same_programs_as_base flag → check does not apply.
    case = _case(test_type="INV", base="b", expect_same_outcome_as_base=True)
    base = Trace(case_id="b", query="q", spans=[], final_text="SNAP",
                 citations={"S1": _cite("SNAP", "u1")}, outcome="answered")
    variant = Trace(case_id="c", query="q2", spans=[], final_text="WIC",
                    citations={"S1": _cite("WIC", "u2")}, outcome="answered")
    assert check_metamorphic_programs(variant, base, case) is None
