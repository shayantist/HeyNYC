from __future__ import annotations

from types import SimpleNamespace

from heynyc.eval.bench import BenchRow, bench_summary, render_bench, run_bench
from heynyc.eval.cases import EvalCase


def _report(pairs):
    """A duck-typed GateReport: bench_summary only reads .reports and each item's .case_id/.passed."""
    return SimpleNamespace(
        reports=[SimpleNamespace(case_id=cid, passed=passed) for cid, passed in pairs]
    )


# --- bench_summary (pure) -------------------------------------------------

def test_bench_summary_counts_overall_and_safety():
    report = _report([("a", True), ("b", False), ("c", True)])
    # overall: 2 of 3 passed. safety subset {b, c}: only c passed → 1 of 2.
    assert bench_summary(report, {"b", "c"}) == (2, 3, 1, 2)


def test_bench_summary_with_no_safety_cases():
    report = _report([("a", True), ("b", True)])
    assert bench_summary(report, set()) == (2, 2, 0, 0)


# --- render_bench ---------------------------------------------------------

def test_render_bench_lists_every_model_and_marks_errors():
    rows = [
        BenchRow(model="gpt-5", report=_report([("a", True), ("b", False)])),
        BenchRow(model="claude-sonnet", report=None, error="RuntimeError: boom"),
    ]
    out = render_bench(rows, safety_case_ids={"b"})
    assert "gpt-5" in out
    assert "claude-sonnet" in out
    assert "ERROR" in out          # the errored row is flagged
    assert "boom" in out           # ...with its error text
    assert "1/2" in out            # gpt-5 overall passed count is surfaced


# --- run_bench continues past a failing model -----------------------------

async def test_run_bench_isolates_a_failing_model(monkeypatch):
    from heynyc.eval import bench as bench_mod

    class _FakeAgent:
        def __init__(self, registry, model=None, index=None):
            if model == "bad-model":
                raise RuntimeError("no such model")
            self.model = model

        async def run(self, query, reminders=None):
            from heynyc.core.agent import AgentResult

            return AgentResult(text="I couldn't find that; try 311.", citations={}, tool_calls_made=[])

    monkeypatch.setattr(bench_mod, "Agent", _FakeAgent)

    case = EvalCase(id="c", module="m", query="q", abstain=True)
    rows = await bench_mod.run_bench(
        ["good-1", "bad-model", "good-2"],
        registry=None, retriever=None, cases=[case], reminders=None,
    )
    # every requested model gets a row, in order — one bad model never aborts the rest.
    assert [r.model for r in rows] == ["good-1", "bad-model", "good-2"]
    bad = next(r for r in rows if r.model == "bad-model")
    assert bad.report is None
    assert "RuntimeError" in bad.error
    for good in (r for r in rows if r.model != "bad-model"):
        assert good.report is not None and good.error is None


# --- candidate cost tracking ----------------------------------------------

def test_candidate_cost_sums_usage_across_cases():
    from heynyc.eval.bench import _candidate_cost
    from heynyc.eval.runner import CaseResult

    results = [
        CaseResult(case=EvalCase(id="a", module="m", query="q"), usage={"input_tokens": 1000, "output_tokens": 500}),
        CaseResult(case=EvalCase(id="b", module="m", query="q"), usage={"input_tokens": 2000, "output_tokens": 800}),
        CaseResult(case=EvalCase(id="c", module="m", query="q"), usage={}),  # a case the agent never billed
    ]
    cost, in_tok, out_tok = _candidate_cost("gpt-4o-mini", results)
    assert in_tok == 3000 and out_tok == 1300   # summed across cases, missing usage counts as 0
    assert cost > 0                              # a litellm-priced model yields a real cost floor


def test_candidate_cost_is_zero_for_an_unpriceable_model():
    from heynyc.eval.bench import _candidate_cost
    from heynyc.eval.runner import CaseResult

    r = [CaseResult(case=EvalCase(id="a", module="m", query="q"), usage={"input_tokens": 10, "output_tokens": 5})]
    cost, in_tok, out_tok = _candidate_cost("some/model-litellm-cannot-price", r)
    assert (in_tok, out_tok) == (10, 5)
    assert cost == 0.0                           # never a fabricated number when pricing is unknown


def test_render_bench_surfaces_candidate_cost():
    rows = [BenchRow(model="gpt-5", report=_report([("a", True)]), cost_usd=0.0123)]
    out = render_bench(rows, safety_case_ids=set())
    assert "$0.0123" in out
