"""Run production and PydanticAI through the same HeyNYC eval route."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from heynyc.__main__ import _default_reminders, _load_retriever
from heynyc.core import config
from heynyc.core.agent import Agent
from heynyc.core.nli import PromptedNLI
from heynyc.core.pydantic_runtime import (
    _uses_openai_responses,
    build_runtime,
)
from heynyc.core.pydantic_runtime import (
    configured_model as _comparison_model,
)
from heynyc.core.registry import Registry
from heynyc.core.telemetry import priced_cost_usd
from heynyc.eval.cases import load_cases, select_cases
from heynyc.eval.report import evaluate, write_run
from heynyc.eval.runner import (
    PydanticEvalAgent,
    PydanticEvalConversation,
    merge_eval_results,
    run_all,
)
from heynyc.modules.advisories.tools import current_awareness

_PydanticEvalConversation = PydanticEvalConversation
_merge_results = merge_eval_results


def _runtime_route(arm: str, model: str) -> str:
    if arm == "production":
        if _uses_openai_responses(model):
            return f"litellm:openai-responses-bridge:{model}"
        return f"litellm.acompletion:{model}"
    if model.startswith("openai/"):
        if _uses_openai_responses(model):
            return f"pydantic-ai:openai-responses:{model.removeprefix('openai/')}"
        return f"pydantic-ai:openai-chat:{model.removeprefix('openai/')}"
    return f"pydantic-ai:{model.replace('/', ':', 1)}"


def summarize_arm(
    arm: str,
    model: str,
    results: list[Any],
    report: Any,
) -> dict[str, Any]:
    turns = [
        turn
        for result in results
        for turn in (getattr(result, "turn_results", None) or ())
    ]
    usages = [getattr(turn, "usage", {}) or {} for turn in turns]
    mechanical_passed = getattr(
        report, "mechanical_passed_count", report.passed_count
    )
    qualitative_pending = getattr(report, "qualitative_pending_count", 0)
    promotion_ready = getattr(
        report, "promotion_ready_count", report.passed_count
    )
    promotion_gate_passed = getattr(
        report, "promotion_ready", promotion_ready == report.total
    )
    total_cost = 0.0
    priced = True
    for usage in usages:
        cost = usage.get("cost_usd")
        if cost is None:
            cost = priced_cost_usd(
                model,
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
                int(
                    usage.get(
                        "cached_input_tokens",
                        usage.get("answer_cached_input_tokens", 0),
                    )
                    or 0
                ),
            )
        if cost is None:
            priced = False
        else:
            total_cost += float(cost)
    return {
        "arm": arm,
        "runtime": "Agent" if arm == "production" else "PydanticRuntimeAdapter",
        "answer_model": model,
        "runtime_route": _runtime_route(arm, model),
        "case_ids": [result.case.id for result in results],
        "passed": report.passed_count,
        "mechanical_passed": mechanical_passed,
        "qualitative_pending": qualitative_pending,
        "promotion_ready": promotion_ready,
        "promotion_gate_passed": promotion_gate_passed,
        "total": report.total,
        "input_tokens": sum(int(u.get("input_tokens", 0) or 0) for u in usages),
        "output_tokens": sum(int(u.get("output_tokens", 0) or 0) for u in usages),
        "cached_input_tokens": sum(
            int(
                u.get(
                    "cached_input_tokens",
                    u.get("answer_cached_input_tokens", 0),
                )
                or 0
            )
            for u in usages
        ),
        "request_count": sum(
            int(u.get("requests", u.get("n_model_calls", 0)) or 0)
            for u in usages
        ),
        "model_call_count": sum(
            int(u.get("n_model_calls", u.get("requests", 0)) or 0)
            for u in usages
        ),
        "tool_call_count": sum(
            int(u.get("n_tool_calls", u.get("tool_calls", 0)) or 0)
            for u in usages
        ),
        "capability_ids": sorted(
            {
                str(capability)
                for usage in usages
                for capability in (
                    *(usage.get("capabilities_used") or ()),
                    *(usage.get("scope_modules") or ()),
                )
            }
        ),
        "latency_ms": sum(float(u.get("latency_ms", 0) or 0) for u in usages),
        "cost_usd": total_cost if priced else None,
        "cost_status": "priced" if priced else "unpriced",
        "error_count": sum(1 for result in results if result.error),
        "fact_confirmation_policy": (
            "auto_confirm_runtime_generated_fact_reviews_only"
            if arm == "pydantic_ai"
            else "not_applicable"
        ),
        "reasoning_effort": config.HEYNYC_REASONING_EFFORT,
    }


def _write_turn_artifacts(directory: Path, results: list[Any]) -> None:
    """Persist every turn so qualitative review cannot hide an earlier failure."""
    cases = []
    for result in results:
        turns = []
        prompts = getattr(result.case, "turns", ())
        started = getattr(result, "turn_started_at", ())
        for index, turn in enumerate(getattr(result, "turn_results", ())):
            turns.append(
                {
                    "turn": index + 1,
                    "started_at": started[index] if index < len(started) else None,
                    "resident_message": prompts[index] if index < len(prompts) else None,
                    "text": getattr(turn, "text", ""),
                    "status": getattr(turn, "status", "success"),
                    "tool_calls": getattr(turn, "tool_calls_made", []),
                    "citations": getattr(turn, "citations", {}),
                    "messages": getattr(turn, "messages", []),
                    "usage": getattr(turn, "usage", {}),
                }
            )
        cases.append({
            "case_id": result.case.id,
            "error": result.error,
            "diagnostics": getattr(result, "diagnostics", {}),
            "turns": turns,
        })
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "turns.json").write_text(
        json.dumps({"cases": cases}, indent=2, default=str)
    )


async def run_arms(
    factories: dict[str, Any],
    cases: list[Any],
    reminders: list[str],
    out_dir: Path,
    model: str,
    *,
    arm_order: tuple[str, ...] | None = None,
    parallel: bool = False,
    structured_grounding: bool = False,
    semantic_verifier_model: str | None = None,
) -> dict[str, Any]:
    order = arm_order or tuple(factories)
    receipt = {
        "case_ids": [case.id for case in cases],
        "arm_order": list(order),
        "parallel": parallel,
        "structured_grounding": structured_grounding,
        "semantic_verifier_model": semantic_verifier_model,
        "performance_comparison_valid": not parallel and len(order) == 2,
        "arms": [],
    }

    async def run_arm(arm: str) -> dict[str, Any]:
        factory = factories[arm]
        results = await run_all(factory, cases, reminders=reminders)
        report = await evaluate(results)
        summary = summarize_arm(arm, model, results, report)
        write_run(out_dir / arm, report, metadata=summary)
        _write_turn_artifacts(out_dir / arm, results)
        return summary

    if parallel:
        receipt["arms"] = list(await asyncio.gather(*(run_arm(arm) for arm in order)))
    else:
        for arm in order:
            receipt["arms"].append(await run_arm(arm))
    (out_dir / "comparison.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def build_factories(
    registry: Registry,
    retriever: Any,
    model: str,
    *,
    structured_grounding: bool = False,
    semantic_verifier: Any = None,
) -> dict[str, Any]:
    candidate_kwargs = {
        "model": _comparison_model(model),
        "answer_model_route": model,
        "index": retriever,
        "use_module_capabilities": True,
        "current_awareness": current_awareness,
    }
    if structured_grounding:
        candidate_kwargs["structured_grounding"] = True
    if semantic_verifier is not None:
        candidate_kwargs["semantic_verifier"] = semantic_verifier
    return {
        "production": lambda: Agent(
            registry,
            model=model,
            index=retriever,
            notify_awareness=current_awareness,
            scope_gate=True,
        ),
        "pydantic_ai": lambda: PydanticEvalAgent(
            build_runtime(registry, **candidate_kwargs)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--module")
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--tag", dest="tags", action="append", default=[])
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", dest="run_all_cases", action="store_true")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run arms concurrently; latency and cache comparisons are not valid",
    )
    parser.add_argument(
        "--arm",
        choices=("production", "pydantic_ai"),
        help="Run only one arm for a focused probe",
    )
    parser.add_argument(
        "--structured-grounding",
        action="store_true",
        help="Use typed grounded answer blocks in the PydanticAI candidate",
    )
    parser.add_argument(
        "--semantic-verifier-model",
        help="Candidate-only claim verifier model; requires --structured-grounding",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.structured_grounding and args.arm == "production":
        print("--structured-grounding only applies to the pydantic_ai arm.")
        return 2
    if args.semantic_verifier_model and not args.structured_grounding:
        print("--semantic-verifier-model requires --structured-grounding.")
        return 2
    registry = Registry.discover(
        config.MODULES_DIR,
        config.BASE_ALLOWLIST,
        config.NEWS_ALLOWLIST,
    )
    cases = load_cases(registry)
    if not (
        args.run_all_cases
        or args.module
        or args.case_ids
        or args.tags
        or args.sample
    ):
        print(
            f"Refusing an unselective {len(cases)}-case live A/B. Pass --all to "
            "confirm, or select with --module, --case, --tag, or --sample."
        )
        return 2
    cases = select_cases(
        cases,
        module=args.module,
        case_ids=args.case_ids or None,
        tags=args.tags or None,
        sample=args.sample,
        seed=args.seed,
    )
    if not cases:
        print("No eval cases found for that selection.")
        return 2

    retriever = _load_retriever(required=False)
    model = config.HEYNYC_MODEL
    semantic_verifier = (
        PromptedNLI(model=args.semantic_verifier_model)
        if args.semantic_verifier_model
        else None
    )
    factories = build_factories(
        registry,
        retriever,
        model,
        structured_grounding=args.structured_grounding,
        semantic_verifier=semantic_verifier,
    )
    if args.arm:
        factories = {args.arm: factories[args.arm]}
    out_dir = Path(args.out) if args.out else (
        config.HEYNYC_DATA_DIR
        / "eval"
        / datetime.now(timezone.utc).strftime("ab-%Y%m%dT%H%M%SZ")
    )
    receipt = await run_arms(
        factories,
        cases,
        _default_reminders(),
        out_dir,
        model,
        arm_order=(
            tuple(factories)
            if args.arm
            else (
                ("pydantic_ai", "production")
                if args.seed % 2
                else ("production", "pydantic_ai")
            )
        ),
        parallel=args.parallel,
        structured_grounding=args.structured_grounding,
        semantic_verifier_model=args.semantic_verifier_model,
    )
    print(json.dumps(receipt, indent=2))
    print(f"Comparison written to {out_dir}")
    return int(any(arm["passed"] != arm["total"] for arm in receipt["arms"]))


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
