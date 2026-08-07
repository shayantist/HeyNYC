from __future__ import annotations

from types import SimpleNamespace

from heynyc.eval.bench import BenchRow, bench_summary, render_bench
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

def test_build_eval_agent_uses_configured_runtime(monkeypatch):
    from heynyc.core import config
    from heynyc.eval import bench as bench_mod
    from heynyc.eval.runner import PydanticEvalAgent

    calls = []

    def fake_pydantic(registry, *, model, index, current_awareness):
        calls.append((registry, model, index, current_awareness))
        return "pydantic-agent"

    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "pydantic")
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_configured_runtime",
        fake_pydantic,
    )

    agent = bench_mod.build_eval_agent("registry", "model", "index")

    assert isinstance(agent, PydanticEvalAgent)
    assert agent.runtime == "pydantic-agent"
    assert calls[0][:3] == ("registry", "model", "index")


def test_build_eval_agent_keeps_legacy_rollback(monkeypatch):
    from heynyc.core import config
    from heynyc.eval import bench as bench_mod

    class FakeLegacy:
        def __init__(self, registry, **kwargs):
            self.registry = registry
            self.kwargs = kwargs

    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "legacy")
    monkeypatch.setattr(bench_mod, "Agent", FakeLegacy)

    agent = bench_mod.build_eval_agent("registry", "model", "index")

    assert agent.registry == "registry"
    assert agent.kwargs["model"] == "model"
    assert agent.kwargs["index"] == "index"
    assert agent.kwargs["scope_gate"] is True


async def test_run_bench_isolates_a_failing_model(monkeypatch):
    from heynyc.eval import bench as bench_mod

    class _FakeAgent:
        def __init__(self, registry, model=None, index=None, scope_gate=False):
            if model == "bad-model":
                raise RuntimeError("no such model")
            self.model = model

        async def run(self, query, reminders=None):
            from heynyc.core.agent import AgentResult

            return AgentResult(text="I couldn't find that; try 311.", citations={}, tool_calls_made=[])

    monkeypatch.setattr(
        bench_mod,
        "build_eval_agent",
        lambda registry, model, retriever: _FakeAgent(
            registry, model=model, index=retriever, scope_gate=True
        ),
    )

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


def test_candidate_cost_is_explicit_for_an_unpriceable_model():
    from heynyc.eval.bench import _candidate_cost
    from heynyc.eval.runner import CaseResult

    r = [CaseResult(case=EvalCase(id="a", module="m", query="q"), usage={"input_tokens": 10, "output_tokens": 5})]
    cost, in_tok, out_tok = _candidate_cost("some/model-litellm-cannot-price", r)
    assert (in_tok, out_tok) == (10, 5)
    assert cost is None                          # never present unknown pricing as free


def test_candidate_cost_uses_per_turn_multi_model_cost():
    from heynyc.eval.bench import _candidate_cost
    from heynyc.eval.runner import CaseResult

    results = [CaseResult(
        case=EvalCase(id="a", module="m", query="q"),
        usage={"input_tokens": 15, "output_tokens": 3, "cost_usd": 0.0042},
    )]

    cost, _, _ = _candidate_cost("gpt-4o-mini", results)

    assert cost == 0.0042


def test_render_bench_surfaces_candidate_cost():
    rows = [BenchRow(model="gpt-5", report=_report([("a", True)]), cost_usd=0.0123)]
    out = render_bench(rows, safety_case_ids=set())
    assert "$0.0123" in out


def test_render_bench_labels_unpriceable_candidate():
    rows = [BenchRow(model="unknown", report=_report([("a", True)]), cost_usd=None)]

    assert "UNPRICED" in render_bench(rows, safety_case_ids=set())


def test_render_bench_surfaces_scope_model_tokens_and_latency():
    rows = [BenchRow(
        model="answer/model", report=_report([("a", True)]), cost_usd=0.01,
        input_tokens=100, output_tokens=20, scope_model="scope/model",
        scope_input_tokens=10, scope_output_tokens=2, scope_time_ms=15.5,
    )]

    out = render_bench(rows, safety_case_ids=set())

    assert "tokens 100/20" in out
    assert "scope scope/model 10/2 tokens 15.5ms" in out


# --- The bare-baseline lane (RULED 2026-07-20, built 2026-07-22): the resident's-ChatGPT-tab
# approximation, the case query sent RAW to a frontier model with no HeyNYC harness ---

async def test_bare_baseline_sends_raw_query_with_no_harness():
    from types import SimpleNamespace

    from heynyc.eval.bench import run_bare_baseline

    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        content = f"raw answer to {kwargs['messages'][0]['content']}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    cases = [SimpleNamespace(id="c1", query="where is the nearest food pantry?")]
    answers = await run_bare_baseline("some/model", cases, completion=fake_completion)

    assert answers == {"c1": "raw answer to where is the nearest food pantry?"}
    assert calls[0]["model"] == "some/model"
    assert calls[0]["messages"] == [
        {"role": "user", "content": "where is the nearest food pantry?"}
    ]
    assert "tools" not in calls[0]  # no harness: no tools, no system prompt, no grounding


async def test_bare_baseline_captures_per_case_errors_without_aborting():
    from types import SimpleNamespace

    from heynyc.eval.bench import run_bare_baseline

    async def boom(**kwargs):
        raise RuntimeError("provider down")

    cases = [SimpleNamespace(id="c1", query="q1"), SimpleNamespace(id="c2", query="q2")]
    answers = await run_bare_baseline("m", cases, completion=boom)
    assert set(answers) == {"c1", "c2"}
    assert all("baseline error" in a for a in answers.values())
