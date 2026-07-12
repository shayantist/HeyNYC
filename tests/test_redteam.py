"""Offline TDD for the reusable red-team harness (heynyc/eval/redteam.py).

Everything here runs with a MOCK candidate and a MOCK grader: no model, no network, no spend.
The load-bearing properties under test (Gap 1: "the safety proof is a method, not evidence at
scale, and it's self-graded"):

  1. the harness routes each frozen case through the candidate (injected/mocked) and captures it;
  2. the INDEPENDENT grader (not the candidate) produces the SAFE/FAIL verdict;
  3. the candidate can NEVER be its own grader (a same-family grader is refused, at construction
     AND at run time);
  4. reconciliation flags a candidate/grader disagreement and the independent grader GOVERNS;
  5. the report aggregates per category, and the frozen suite parses + covers all 8 categories.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from heynyc.core.agent import AgentResult
from heynyc.eval.redteam import (
    CATEGORIES,
    RedTeamCase,
    RedTeamCaseResult,
    RedTeamGrader,
    RedTeamReport,
    RedTeamVerdict,
    default_suite_path,
    load_suite,
    model_family,
    reconcile,
    run_redteam,
    same_family,
)


# --- helpers ----------------------------------------------------------------

def _case(cid="MC01", category="MC", query="q", framing="owner", language="en"):
    return RedTeamCase(
        id=cid, category=category, framing=framing, language=language, query=query,
        adversarial_intent="intent", safety_criterion="criterion",
    )


class _FakeAgent:
    """A candidate stand-in. Records the queries it saw so a test can prove it ran."""

    seen: list[str] = []

    def __init__(self, answer="A benign, grounded answer.", citations=None):
        self.answer = answer
        self.citations = citations or {}

    async def run(self, query, reminders=None):
        _FakeAgent.seen.append(query)
        return AgentResult(text=self.answer, citations=self.citations, tool_calls_made=[])


class _MockGrader:
    """A grader stand-in: returns a canned verdict per case_id. It is a DIFFERENT object than the
    candidate, so a verdict coming from here proves the grader (not the candidate) scored."""

    def __init__(self, verdicts: dict, grader_model="anthropic/claude-sonnet-4-6"):
        self._verdicts = verdicts
        self.grader_model = grader_model
        self.candidate_model = "openai/gpt-5-mini"

    async def grade(self, cr, case) -> RedTeamVerdict:
        v = self._verdicts.get(case.id, "SAFE")
        return RedTeamVerdict(case_id=case.id, verdict=v, reason=f"mock {v}",
                              offending_quote="", grader_model=self.grader_model)


# --- model family separation (candidate can never self-grade) ---------------

def test_model_family_reads_the_provider_prefix():
    assert model_family("openai/gpt-5-mini") == "openai"
    assert model_family("anthropic/claude-sonnet-4-6") == "anthropic"
    assert model_family("ollama/qwen2.5:27b-instruct") == "ollama"
    # bare names (no provider prefix) are inferred by name
    assert model_family("gpt-4o-mini") == "openai"
    assert model_family("claude-3-5-haiku") == "anthropic"


def test_same_family_true_only_within_a_provider():
    assert same_family("openai/gpt-5-mini", "openai/gpt-4o-mini") is True
    assert same_family("openai/gpt-5-mini", "openai/gpt-5-mini") is True
    assert same_family("openai/gpt-5-mini", "anthropic/claude-sonnet-4-6") is False
    assert same_family("ollama/qwen2.5:27b", "openai/gpt-4o-mini") is False


def test_grader_refuses_to_be_constructed_in_the_candidate_family():
    # The whole point of Gap 1: the candidate must never grade its own output.
    with pytest.raises(ValueError):
        RedTeamGrader(grader_model="openai/gpt-5-mini", candidate_model="openai/gpt-5-mini")
    with pytest.raises(ValueError):  # same provider, different size still self-enhancement-biased
        RedTeamGrader(grader_model="openai/gpt-4o-mini", candidate_model="openai/gpt-5-mini")
    # cross-family is allowed
    g = RedTeamGrader(grader_model="anthropic/claude-sonnet-4-6", candidate_model="openai/gpt-5-mini")
    assert g.grader_model == "anthropic/claude-sonnet-4-6"


# --- the grader (not the candidate) renders the verdict ---------------------

async def test_grader_verdict_comes_from_the_grader_completion_not_the_candidate():
    # The candidate's own text says nothing about a verdict; the grader's completion decides.
    async def grader_complete(prompt):
        return '{"verdict": "FAIL", "reason": "wrong court", "offending_quote": "Third Department"}'

    grader = RedTeamGrader(
        grader_model="anthropic/claude-sonnet-4-6",
        candidate_model="openai/gpt-5-mini",
        complete_fn=grader_complete,
    )
    cr = SimpleNamespace(case=_case("MC03"), text="Landlords cannot refuse CityFHEPS.",
                         citations={}, error=None)
    verdict = await grader.grade(cr, _case("MC03"))
    assert verdict.verdict == "FAIL"
    assert verdict.offending_quote == "Third Department"
    assert verdict.grader_model == "anthropic/claude-sonnet-4-6"  # stamped by the grader, not candidate


async def test_grader_defaults_to_fail_when_output_is_unparseable():
    # Strict adversarial rule: default to FAIL when uncertain.
    async def garbage(prompt):
        return "I think this is probably fine, no JSON here"

    grader = RedTeamGrader("anthropic/claude-sonnet-4-6", "openai/gpt-5-mini", complete_fn=garbage)
    verdict = await grader.grade(SimpleNamespace(case=_case(), text="hi", citations={}, error=None), _case())
    assert verdict.verdict == "FAIL"


async def test_grader_marks_a_candidate_crash_as_fail():
    async def unused(prompt):  # pragma: no cover - must not be called on an errored case
        raise AssertionError("grader completion should not run for an errored candidate")

    grader = RedTeamGrader("anthropic/claude-sonnet-4-6", "openai/gpt-5-mini", complete_fn=unused)
    cr = SimpleNamespace(case=_case(), text="", citations={}, error="agent blew up")
    verdict = await grader.grade(cr, _case())
    assert verdict.verdict == "FAIL"


# --- the harness routes every case and lets the grader score ----------------

async def test_run_redteam_routes_each_case_and_scores_with_the_grader():
    _FakeAgent.seen = []
    suite = [_case("MC01", "MC", "owner q"), _case("PI01", "PI", "inject q"),
             _case("ES01", "ES", "spanish q", language="es")]
    grader = _MockGrader({"PI01": "FAIL"})  # grader fails one, passes the rest

    report = await run_redteam(
        candidate_model="openai/gpt-5-mini",
        suite=suite,
        grader=grader,
        agent_factory=lambda: _FakeAgent(),
    )
    # every case was routed through the candidate
    assert _FakeAgent.seen == ["owner q", "inject q", "spanish q"]
    assert len(report.rows) == 3
    by_id = {r.case_id: r for r in report.rows}
    # verdicts came from the grader, not the (benign) candidate text
    assert by_id["PI01"].verdict == "FAIL"
    assert by_id["MC01"].verdict == "SAFE"
    assert by_id["ES01"].verdict == "SAFE"


async def test_run_redteam_refuses_a_candidate_in_the_grader_family():
    # Harness-level guard (defense in depth beyond the grader constructor): even a validly-built
    # grader must not be pointed at a candidate that shares its family.
    grader = RedTeamGrader("anthropic/claude-sonnet-4-6", "openai/gpt-5-mini",
                           complete_fn=lambda p: None)
    with pytest.raises(ValueError):
        await run_redteam(
            candidate_model="anthropic/claude-sonnet-4-6",  # same family as the grader
            suite=[_case()],
            grader=grader,
            agent_factory=lambda: _FakeAgent(),
        )


# --- reconciliation: the independent grader GOVERNS, disagreements are flagged

def test_reconcile_independent_governs_and_flags_disagreement():
    indep = RedTeamVerdict("MC03", "FAIL", "wrong court")
    second = RedTeamVerdict("MC03", "SAFE", "looked fine to me")
    final, disagreement = reconcile(indep, second)
    assert final == "FAIL"          # the independent grader governs
    assert disagreement is True     # the candidate/second-opinion overclaim is surfaced


def test_reconcile_no_disagreement_when_they_agree_or_no_second_opinion():
    assert reconcile(RedTeamVerdict("x", "SAFE"), RedTeamVerdict("x", "SAFE")) == ("SAFE", False)
    assert reconcile(RedTeamVerdict("x", "SAFE"), None) == ("SAFE", False)


async def test_run_redteam_flags_candidate_self_grade_disagreement_but_grader_governs():
    # The "we caught our own overclaim" story, made structural: the candidate self-assesses SAFE
    # (advisory), the independent grader says FAIL (governing). Final = FAIL, flagged.
    grader = _MockGrader({"MC03": "FAIL"})
    self_opinion = _MockGrader({"MC03": "SAFE"})  # stands in for the candidate self-assessment
    report = await run_redteam(
        candidate_model="openai/gpt-5-mini",
        suite=[_case("MC03", "MC")],
        grader=grader,
        second_opinion=self_opinion,
        agent_factory=lambda: _FakeAgent(),
    )
    row = report.rows[0]
    assert row.verdict == "FAIL"            # independent grader governs
    assert row.independent_verdict == "FAIL"
    assert row.second_verdict == "SAFE"     # the advisory (candidate) opinion is retained...
    assert row.disagreement is True         # ...and the disagreement is flagged, not hidden
    assert report.disagreements() == [row]


# --- the report aggregates per category -------------------------------------

def _row(cid, cat, verdict, disagreement=False):
    return RedTeamCaseResult(
        case_id=cid, category=cat, framing="f", language="en", verdict=verdict,
        reason="", offending_quote="", grader_model="anthropic/claude-sonnet-4-6",
        disagreement=disagreement, independent_verdict=verdict, second_verdict="",
    )


def test_report_aggregates_per_category_and_totals():
    report = RedTeamReport(
        rows=[
            _row("MC01", "MC", "SAFE"), _row("MC02", "MC", "FAIL"),
            _row("PI01", "PI", "SAFE"), _row("PI02", "PI", "SAFE"),
            _row("ES01", "ES", "FAIL", disagreement=True),
        ],
        candidate_model="openai/gpt-5-mini",
        grader_model="anthropic/claude-sonnet-4-6",
    )
    per = report.per_category()
    assert per["MC"] == {"safe": 1, "fail": 1, "total": 2, "disagreements": 0}
    assert per["PI"] == {"safe": 2, "fail": 0, "total": 2, "disagreements": 0}
    assert per["ES"] == {"safe": 0, "fail": 1, "total": 1, "disagreements": 1}
    t = report.totals()
    assert t["total"] == 5 and t["safe"] == 3 and t["fail"] == 2 and t["disagreements"] == 1
    out = report.render()
    assert "MC" in out and "PI" in out and "ES" in out
    assert "openai/gpt-5-mini" in out and "anthropic/claude-sonnet-4-6" in out


# --- the frozen suite parses and covers all 8 categories --------------------

def test_frozen_suite_parses_and_covers_all_eight_categories():
    suite = load_suite()
    assert len(suite) >= 137  # at least the v1 set, expanded
    cats = {c.category for c in suite}
    assert cats == set(CATEGORIES), f"missing categories: {set(CATEGORIES) - cats}"
    # ids are unique
    ids = [c.id for c in suite]
    assert len(ids) == len(set(ids)), "duplicate case ids in the frozen suite"
    # every case is a PROMPT with an intent and a safety criterion, never a canned answer
    for c in suite:
        assert c.query.strip(), f"{c.id} has an empty query"
        assert c.adversarial_intent.strip(), f"{c.id} has no adversarial_intent"
        assert c.safety_criterion.strip(), f"{c.id} has no safety_criterion"
        assert not hasattr(c, "answer") and not hasattr(c, "expected_answer")


def test_frozen_suite_grows_the_two_undersampled_categories():
    # red-team-v1 flagged Spanish (ES) and prompt-injection (PI) as under-sampled; v2 must expand
    # both beyond their v1 counts (ES 18, PI 18).
    suite = load_suite()
    from collections import Counter

    counts = Counter(c.category for c in suite)
    assert counts["ES"] > 18, f"ES not expanded: {counts['ES']}"
    assert counts["PI"] > 18, f"PI not expanded: {counts['PI']}"
    # every category is at least as large as its v1 count
    v1 = {"MC": 20, "PI": 18, "OS": 18, "FP": 16, "HS": 16, "PII": 15, "CIT": 16, "ES": 18}
    for cat, n in v1.items():
        assert counts[cat] >= n, f"{cat} shrank vs v1: {counts[cat]} < {n}"


def test_default_suite_path_points_at_the_shipped_yaml():
    p = default_suite_path()
    assert p.name.endswith(".yaml") and p.exists()
