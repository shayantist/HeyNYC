"""Owner-gated live entrypoint for the adversarial red-team.

This is a THIN orchestration over the shared eval machinery, it reinvents nothing:

  • the suite is `cases.load_redteam_cases()` (205 attack prompts as EvalCases, each carrying a
    `safety_criterion` so the shared judge grades them with the strict adversarial rubric);
  • the runner + gate are `bench.run_bench` / `report.evaluate`, the SAME path `heynyc eval` and
    `heynyc bench` use, so red-team answers also pick up the deterministic grounding floor,
    link-liveness, and invariants a standalone harness would have skipped;
  • fresh subscription-agent review is the default; the paid cross-family API judge is available
    only through the explicit `--api-judge` option;
  • the headline view is `bench.render_by_category` (review state plus per-category results).

Standards (see docs/internal/eval/red-team-v2-methodology.md): the 8 categories + strict SAFE/FAIL rubric are
frozen from red-team-v1; harm categories map to the OWASP Top 10 for LLM Applications (2025) and
MLCommons AILuminate; the grader is an "Agent-as-a-Judge" (arXiv 2410.10934) reading the trace.

NOTHING here runs a model on import or in tests; the live run is owner-gated (see __main__ below).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core import config
from .bench import BenchRow, render_by_category, run_bench
from .cases import (
    CATEGORY_NAMES,
    REDTEAM_REVIEW_REQUIRED,
    default_redteam_suite,
    load_redteam_cases,
)
from .judges import (  # re-exported for the package API
    make_api_judge,
    model_family,
    same_family,
)

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


def _load_standalone_env() -> None:
    """Load `.env` before a standalone run reads environment-backed config."""
    from importlib import reload

    from dotenv import load_dotenv

    load_dotenv()
    reload(config)


async def run_redteam(
    candidate_model: str,
    grader_model: Optional[str] = None,
    cases=None,
    registry=None,
    retriever=None,
    reminders: Optional[list[str]] = None,
    now: Optional[datetime] = None,
    out_dir: Optional[str] = None,
    run_metadata: Optional[dict] = None,
    api_judge: bool = False,
) -> BenchRow:
    """Run one candidate through the shared bench and return its single BenchRow."""
    cases = cases if cases is not None else load_redteam_cases()
    if not cases:
        raise ValueError("red-team suite has no cases")
    for case in cases:
        case.utility_criterion = case.utility_criterion or REDTEAM_REVIEW_REQUIRED
    if grader_model is not None and not api_judge:
        raise ValueError("--grader is only valid with --api-judge")
    judge = None
    run_time = now or datetime.now()
    out_dir = out_dir or f".data/redteam/run-{run_time.strftime('%Y%m%dT%H%M%S')}"
    metadata = {
        **(run_metadata or {}),
        "candidate_model": candidate_model,
        "runtime": config.HEYNYC_AGENT_RUNTIME,
        "review_mode": "api-judge" if api_judge else "subscription-agent-pending",
        "run_started_at": run_time.isoformat(),
    }
    if api_judge:
        grader_model = grader_model or config.HEYNYC_JUDGE_MODEL
        judge = make_api_judge(
            grader_model,
            now=now,
            candidate_model=candidate_model,
            require_independent=True,
        )
        metadata["grader_model"] = grader_model
    else:
        metadata.pop("grader_model", None)
    rows = await run_bench(
        [candidate_model], registry, retriever, cases, reminders, judge=judge, out_dir=out_dir,
        run_metadata=metadata,
    )
    return rows[0]


def _main() -> None:  # pragma: no cover - live path; exercised only by the owner, never in tests
    """`python -m heynyc.eval.redteam --model openai/gpt-5-mini` runs the suite live.

    OWNER-GATED: every run spends candidate API calls. Review defaults to a fresh subscription
    agent reading the saved traces. `--api-judge` additionally spends grader API calls and enforces
    a model family different from the candidate."""
    _load_standalone_env()

    import argparse
    import asyncio

    from ..core.registry import Registry

    parser = argparse.ArgumentParser(
        prog="python -m heynyc.eval.redteam",
        description="Run the frozen adversarial red-team suite against a candidate model.",
    )
    parser.add_argument("--model", required=True, help="candidate model id, e.g. openai/gpt-5-mini")
    parser.add_argument(
        "--api-judge",
        action="store_true",
        help="opt in to the paid cross-family API judge; default review is subscription-agent-pending",
    )
    parser.add_argument("--grader", default=None,
                        help="grader model id for --api-judge (must differ from --model family); "
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

    if args.grader and not args.api_judge:
        parser.error("--grader requires --api-judge")
    grader_model = (
        args.grader or config.HEYNYC_JUDGE_MODEL
        if args.api_judge
        else None
    )
    if grader_model and same_family(args.model, grader_model):
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

    review = (
        f"independent grader: {grader_model}"
        if args.api_judge
        else "subscription-agent review pending"
    )
    print(f"Red-teaming {args.model} on {len(cases)} case(s); {review}")
    row = asyncio.run(
        run_redteam(
            candidate_model=args.model,
            grader_model=grader_model,
            api_judge=args.api_judge,
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
