"""Owner-gated live entrypoint for the adversarial red-team.

This is a THIN orchestration over the shared eval machinery, it reinvents nothing:

  • the suite is `cases.load_redteam_cases()` (205 attack prompts as EvalCases, each carrying a
    `safety_criterion` so the shared judge grades them with the strict adversarial rubric);
  • the runner + gate are `bench.run_bench` / `report.evaluate`, the SAME path `heynyc eval` and
    `heynyc bench` use, so red-team answers also pick up the deterministic grounding floor,
    link-liveness, and invariants a standalone harness would have skipped;
  • the grader is `judges.make_api_judge(..., require_independent=True)`, an INDEPENDENT,
    family-separated cross-family judge that STRUCTURALLY refuses to be built in the candidate's own
    model family (the candidate can never grade its own output, OTI-readiness Gap 1's whole point);
  • the headline view is `bench.render_by_category` (per-category SAFE counts, any failure flagged).

Standards (see docs/eval/red-team-v2-methodology.md): the 8 categories + strict SAFE/FAIL rubric are
frozen from red-team-v1; harm categories map to the OWASP Top 10 for LLM Applications (2025) and
MLCommons AILuminate; the grader is an "Agent-as-a-Judge" (arXiv 2410.10934) reading the trace.

NOTHING here runs a model on import or in tests; the live run is owner-gated (see __main__ below).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from .bench import BenchRow, render_by_category, run_bench
from .cases import CATEGORY_NAMES, default_redteam_suite, load_redteam_cases
from .judges import make_api_judge, model_family, same_family  # re-exported for the package API

__all__ = [
    "run_redteam", "render_by_category", "load_redteam_cases", "model_family", "same_family",
    "select_cases",
]


def select_cases(cases, case_ids: list[str]):
    """Select named cases in caller order so a failed subset can be rerun cheaply."""
    by_id = {case.id: case for case in cases}
    unknown = [case_id for case_id in case_ids if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown red-team case id(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in case_ids]


async def run_redteam(
    candidate_model: str,
    grader_model: str,
    cases=None,
    registry=None,
    retriever=None,
    reminders: Optional[list[str]] = None,
    now: Optional[datetime] = None,
    out_dir: Optional[str] = None,
    run_metadata: Optional[dict] = None,
) -> BenchRow:
    """Red-team one candidate: run the adversarial suite through the shared bench, graded by an
    INDEPENDENT cross-family judge. Returns the single BenchRow (its `.report` is the full GateReport).

    Building the judge with `require_independent=True` raises BEFORE any model runs when the grader
    shares the candidate's family, the structural self-grading guard, surfaced loudly rather than
    swallowed as a per-model error."""
    cases = cases if cases is not None else load_redteam_cases()
    judge = make_api_judge(grader_model, now=now, candidate_model=candidate_model, require_independent=True)
    rows = await run_bench(
        [candidate_model], registry, retriever, cases, reminders, judge=judge, out_dir=out_dir,
        run_metadata={
            **(run_metadata or {}),
            "candidate_model": candidate_model,
            "grader_model": grader_model,
            "run_started_at": (now or datetime.now()).isoformat(),
        },
    )
    return rows[0]


def _main() -> None:  # pragma: no cover - live path; exercised only by the owner, never in tests
    """`python -m heynyc.eval.redteam --model openai/gpt-5-mini` runs the suite live.

    OWNER-GATED: it spends the candidate + grader API keys. The grader defaults to
    config.HEYNYC_JUDGE_MODEL and MUST be a different family than the candidate (the judge refuses to
    be built otherwise), which is exactly the Gap-1 guarantee."""
    from dotenv import load_dotenv
    load_dotenv()  # standalone entrypoint: load .env so the candidate + grader API keys are present

    import argparse
    import asyncio

    from ..core import config
    from ..core.registry import Registry

    parser = argparse.ArgumentParser(
        prog="python -m heynyc.eval.redteam",
        description="Run the frozen adversarial red-team suite against a candidate model, scored by "
                    "an independent cross-family grader.",
    )
    parser.add_argument("--model", required=True, help="candidate model id, e.g. openai/gpt-5-mini")
    parser.add_argument("--grader", default=None,
                        help="grader model id (must be a different family than --model); "
                             "default: HEYNYC_JUDGE_MODEL")
    parser.add_argument("--suite", default=None, help="path to a suite YAML (default: the shipped one)")
    parser.add_argument("--out", default=None, help="directory to write per-model report.json/.txt into")
    parser.add_argument("--category", default=None,
                        help=f"only run one category ({'/'.join(CATEGORY_NAMES)})")
    parser.add_argument(
        "--case", action="append", default=[],
        help="run one case id; repeat the option for a focused subset",
    )
    args = parser.parse_args()

    grader_model = args.grader or config.HEYNYC_JUDGE_MODEL
    if same_family(args.model, grader_model):
        raise SystemExit(
            f"grader '{grader_model}' shares family '{model_family(grader_model)}' with candidate "
            f"'{args.model}', that is self-grading. Pass --grader with a different model family."
        )

    cases = load_redteam_cases(Path(args.suite) if args.suite else default_redteam_suite())
    if args.category:
        cases = [c for c in cases if c.redteam_category == args.category]
        if not cases:
            raise SystemExit(f"no cases in category {args.category!r}")
    if args.case:
        try:
            cases = select_cases(cases, args.case)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    # Do not load the local semantic index here. It can initialize an embedding model on the
    # owner's laptop. Live red-teams use official web retrieval unless a caller explicitly injects
    # an already-running retriever into run_redteam().
    retriever = None

    print(f"Red-teaming {args.model} on {len(cases)} case(s); independent grader: {grader_model}")
    row = asyncio.run(
        run_redteam(
            candidate_model=args.model,
            grader_model=grader_model,
            cases=cases,
            registry=registry,
            retriever=retriever,
            now=datetime.now(),
            out_dir=args.out,
        )
    )
    print("\n" + render_by_category([row], cases))
    if row.report is not None:
        print("\n" + row.report.render())
    elif row.error:
        raise SystemExit(f"candidate run failed: {row.error}")


if __name__ == "__main__":  # pragma: no cover
    _main()
