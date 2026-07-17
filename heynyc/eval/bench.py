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
from .report import GateReport, evaluate, write_run
from .runner import run_all


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
        buckets: dict[str, list[bool]] = {}
        for r in row.report.reports:
            buckets.setdefault(cat_of.get(r.case_id, "?"), []).append(r.passed)
        op = sum(1 for r in row.report.reports if r.passed)
        ot = len(row.report.reports)
        cost = " | UNPRICED" if row.cost_usd is None else f" | ${row.cost_usd:.4f}"
        scope = (
            f" | scope {row.scope_model} {row.scope_input_tokens}/{row.scope_output_tokens} "
            f"tokens {row.scope_time_ms:.1f}ms"
            if row.scope_model else ""
        )
        lines.append(
            f"  {row.model}: SAFE {op}/{ot} | tokens {row.input_tokens}/{row.output_tokens}"
            f"{scope}{cost}"
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
                return Agent(registry, model=m, index=retriever, scope_gate=True)

            results = await run_all(factory, cases, reminders=reminders)
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
