"""Multi-model eval bench, run the golden cases across several candidate backend models and print a per-model comparison.

When a new model ships, point the bench at it and the incumbents in one shot (`heynyc bench --models a,b,c`)
and read off overall + safety-critical pass rates side by side to decide whether to switch. This reuses the
exact same case set, runner, and gate as `heynyc eval`, the only new axis is "which model answered".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.agent import Agent
from ..core.telemetry import priced_cost_usd
from .report import GateReport, evaluate, progress_writer, write_run
from .runner import PydanticEvalAgent, run_all


def build_eval_agent(registry, model: str, retriever):
    """Build the same selected runtime used by resident-facing channels."""
    from ..core import config
    from ..modules.advisories.tools import current_awareness

    if config.HEYNYC_AGENT_RUNTIME == "pydantic":
        from ..core.pydantic_runtime import build_configured_runtime

        return PydanticEvalAgent(
            build_configured_runtime(
                registry,
                model=model,
                index=retriever,
                stream_model_requests=True,
            )
        )
    return Agent(
        registry,
        model=model,
        index=retriever,
        notify_awareness=current_awareness,
        scope_gate=True,
    )


@dataclass
class BenchRow:
    """One model's result: its gate report, or an error string if that model's run blew up."""
    model: str
    report: Optional[GateReport]
    error: Optional[str] = None
    cost_usd: Optional[float] = 0.0  # None means at least one candidate call was unpriceable
    input_tokens: int = 0
    output_tokens: int = 0
    scope_model: str = ""
    scope_input_tokens: int = 0
    scope_output_tokens: int = 0
    scope_time_ms: float = 0.0


def _candidate_cost(model: str, results) -> tuple[Optional[float], int, int]:
    """Total candidate spend for one model's run: sum every case's token usage, price it once.

    Returns (cost_usd, input_tokens, output_tokens). None means at least one call could not be priced,
    so the run must not be presented as free."""
    results = [
        turn
        for result in results
        for turn in (getattr(result, "turn_results", None) or [result])
    ]
    in_tok = sum(int(r.usage.get("input_tokens", 0)) for r in results)
    out_tok = sum(int(r.usage.get("output_tokens", 0)) for r in results)
    total = 0.0
    for result in results:
        if "cost_usd" in result.usage:
            cost = result.usage["cost_usd"]
        else:
            cost = priced_cost_usd(
                model,
                int(result.usage.get("input_tokens", 0)),
                int(result.usage.get("output_tokens", 0)),
            )
        if cost is None:
            return None, in_tok, out_tok
        total += float(cost)
    return total, in_tok, out_tok


def _scope_metrics(results) -> tuple[str, int, int, float]:
    models = sorted({str(r.usage.get("scope_model") or "") for r in results} - {""})
    return (
        ",".join(models),
        sum(int(r.usage.get("scope_input_tokens", 0) or 0) for r in results),
        sum(int(r.usage.get("scope_output_tokens", 0) or 0) for r in results),
        sum(float(r.usage.get("scope_time_ms", 0.0) or 0.0) for r in results),
    )


def bench_summary(report, safety_case_ids: set) -> tuple[int, int, int, int]:
    """Pure. Return (overall_passed, overall_total, safety_passed, safety_total) from a gate report.

    Reads only report.reports and each element's .case_id / .passed, so it works on the real GateReport
    and on any duck-typed stand-in. safety_passed/total count only reports whose case_id is a safety id.
    """
    overall_total = len(report.reports)
    overall_passed = sum(1 for r in report.reports if r.passed)
    safety = [r for r in report.reports if r.case_id in safety_case_ids]
    safety_total = len(safety)
    safety_passed = sum(1 for r in safety if r.passed)
    return overall_passed, overall_total, safety_passed, safety_total


def render_bench(rows: list[BenchRow], safety_case_ids: set) -> str:
    """A scannable plain-text comparison block: a header plus one line per model, with candidate cost."""
    lines = ["HeyNYC model bench", ""]
    for row in rows:
        if row.error is not None:
            lines.append(f"  {row.model}: ERROR ({row.error})")
            continue
        op, ot, sp, st = bench_summary(row.report, safety_case_ids)
        cost = " | UNPRICED" if row.cost_usd is None else f" | ${row.cost_usd:.4f}"
        scope = (
            f" | scope {row.scope_model} {row.scope_input_tokens}/{row.scope_output_tokens} "
            f"tokens {row.scope_time_ms:.1f}ms"
            if row.scope_model else ""
        )
        lines.append(
            f"  {row.model}: overall {op}/{ot} | safety-critical {sp}/{st} | "
            f"tokens {row.input_tokens}/{row.output_tokens}{scope}{cost}"
        )
    return "\n".join(lines)


def render_by_category(rows: list[BenchRow], cases) -> str:
    """Per-category pass breakdown, one block per model, the red-team's headline view.

    `cases` supplies each case's `redteam_category` (falling back to `harm_category`), so this works on
    the red-team suite or any category-tagged case set. A category with any failure is flagged so a
    single unsafe answer can never hide inside a high pass rate."""
    cat_of = {c.id: (getattr(c, "redteam_category", "") or getattr(c, "harm_category", "") or "?") for c in cases}
    lines = ["HeyNYC red-team, per-category safety", ""]
    for row in rows:
        if row.error is not None:
            lines.append(f"  {row.model}: ERROR ({row.error})")
            continue

        def reviewed_passed(report) -> bool:
            if (
                getattr(report, "qualitative_review_required", False)
                and getattr(report, "qualitative_reviewed", False)
            ):
                return bool(getattr(report, "promotion_ready", False))
            return bool(report.passed)

        buckets: dict[str, list[bool]] = {}
        for r in row.report.reports:
            buckets.setdefault(cat_of.get(r.case_id, "?"), []).append(reviewed_passed(r))
        op = sum(1 for r in row.report.reports if reviewed_passed(r))
        ot = len(row.report.reports)
        pending = sum(
            1
            for r in row.report.reports
            if getattr(r, "qualitative_review_required", False)
            and not getattr(r, "qualitative_reviewed", False)
        )
        cost = " | UNPRICED" if row.cost_usd is None else f" | ${row.cost_usd:.4f}"
        scope = (
            f" | scope {row.scope_model} {row.scope_input_tokens}/{row.scope_output_tokens} "
            f"tokens {row.scope_time_ms:.1f}ms"
            if row.scope_model else ""
        )
        if any(not reviewed_passed(r) for r in row.report.reports) or not ot:
            status = f"FAIL {op}/{ot}"
        elif pending:
            status = f"MECHANICAL {op}/{ot} | REVIEW PENDING {pending}"
        elif op == ot:
            status = f"SAFE {op}/{ot}"
        else:
            status = f"FAIL {op}/{ot}"
        lines.append(
            f"  {row.model}: {status} | tokens {row.input_tokens}/{row.output_tokens}{scope}{cost}"
        )
        for cat in sorted(buckets):
            passed = sum(buckets[cat])
            total = len(buckets[cat])
            flag = "" if passed == total else "  <-- FAIL"
            lines.append(f"      {cat:<4} {passed}/{total}{flag}")
    return "\n".join(lines)


async def run_bench(
    models: list[str],
    registry,
    retriever,
    cases,
    reminders,
    judge=None,
    out_dir: Optional[str] = None,
    run_metadata: Optional[dict] = None,
) -> list[BenchRow]:
    """Run the full case set once per model and collect a BenchRow each. One bad model never aborts the rest.

    Each model gets a fresh-agent factory (state never leaks between cases or models). Any exception from a
    model's run is captured into that row's `.error` and the loop continues. When `out_dir` is given, each
    successful model's report is written to a per-model subdirectory so the raw answers/traces stay inspectable.
    """
    rows: list[BenchRow] = []
    for model in models:
        try:
            def factory(m=model):
                return build_eval_agent(registry, m, retriever)

            # Persist each case as it lands. A stall or a kill on a long paid run used to lose
            # every completed case, since the report was only assembled at the end
            progress = progress_writer(Path(out_dir) / model) if out_dir is not None else None
            results = await run_all(factory, cases, reminders=reminders, on_case=progress)
            report = await evaluate(results, judge=judge)
            if out_dir is not None:
                # write_run(directory, report), one subdir per model keeps raw traces/answers separate.
                write_run(Path(out_dir) / model, report, metadata=run_metadata)
            cost, in_tok, out_tok = _candidate_cost(model, results)
            scope_model, scope_in, scope_out, scope_ms = _scope_metrics(results)
            rows.append(BenchRow(
                model, report, cost_usd=cost, input_tokens=in_tok, output_tokens=out_tok,
                scope_model=scope_model, scope_input_tokens=scope_in,
                scope_output_tokens=scope_out, scope_time_ms=scope_ms,
            ))
        except Exception as e:  # a broken/unauthorized model must not sink the whole comparison
            rows.append(BenchRow(model, None, error=f"{type(e).__name__}: {e}"))
    return rows


async def run_bare_baseline(model: str, cases, *, completion=None) -> dict[str, str]:
    """The resident's-ChatGPT-tab approximation (RULED 2026-07-20): each case query sent RAW
    to a frontier model. No system prompt, no tools, no grounding, no HeyNYC harness, so the
    head-to-head stops being reviewer speculation and becomes columns. Returns case id ->
    answer text; a per-case provider error is captured inline and never aborts the sweep."""
    if completion is None:
        import litellm

        completion = litellm.acompletion
    answers: dict[str, str] = {}
    for case in cases:
        try:
            response = await completion(
                model=model, messages=[{"role": "user", "content": case.query}]
            )
            answers[case.id] = response.choices[0].message.content or ""
        except Exception as e:
            answers[case.id] = f"[baseline error: {type(e).__name__}: {e}]"
    return answers
