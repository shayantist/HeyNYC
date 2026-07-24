"""Run production and PydanticAI through the same HeyNYC eval route."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litellm.main import responses_api_bridge_check
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from heynyc.__main__ import _default_reminders, _load_retriever
from heynyc.core import config
from heynyc.core.agent import Agent, AgentResult
from heynyc.core.registry import Registry
from heynyc.core.telemetry import priced_cost_usd
from heynyc.eval.cases import load_cases, select_cases
from heynyc.eval.report import evaluate, write_run
from heynyc.eval.runner import run_all
from heynyc.modules.advisories.tools import current_awareness
from scripts.pydantic_ai_parity import build_runtime

_ADDITIVE_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "answer_input_tokens",
    "answer_output_tokens",
    "answer_cached_input_tokens",
    "requests",
    "tool_calls",
    "n_model_calls",
    "n_answer_model_calls",
    "n_tool_calls",
    "iterations",
    "latency_ms",
    "model_time_ms",
}


def _merge_results(pending: AgentResult, final: AgentResult) -> AgentResult:
    usage = {**pending.usage, **final.usage}
    for key in _ADDITIVE_USAGE_KEYS:
        if key in pending.usage or key in final.usage:
            usage[key] = (pending.usage.get(key) or 0) + (final.usage.get(key) or 0)
    costs = (pending.usage.get("cost_usd"), final.usage.get("cost_usd"))
    usage["cost_usd"] = (
        sum(costs) if all(isinstance(cost, (int, float)) for cost in costs) else None
    )
    usage["cost_status"] = "priced" if usage["cost_usd"] is not None else "unpriced"
    usage["capabilities_used"] = list(
        dict.fromkeys(
            [
                *(pending.usage.get("capabilities_used") or ()),
                *(final.usage.get("capabilities_used") or ()),
            ]
        )
    )
    return AgentResult(
        text=final.text,
        citations={**pending.citations, **final.citations},
        tool_calls_made=[*pending.tool_calls_made, *final.tool_calls_made],
        iterations=pending.iterations + final.iterations,
        hit_max_iters=pending.hit_max_iters or final.hit_max_iters,
        status=final.status,
        messages=[*pending.messages, *final.messages],
        usage=usage,
    )


class _PydanticEvalConversation:
    """Complete native fact review in evals; never approve an external action."""

    def __init__(self, conversation: Any):
        self.conversation = conversation

    async def send(self, message: str, **kwargs: Any) -> AgentResult:
        pending = await self.conversation.send(message, **kwargs)
        approvals = self.conversation.pending_approvals
        if (
            pending.status != "approval_required"
            or not approvals
            or not all(
                request["tool_name"].startswith("confirm_")
                and request["tool_name"].endswith("_facts")
                for request in approvals.values()
            )
        ):
            return pending
        final = await self.conversation.resume_approvals(
            {call_id: True for call_id in approvals}
        )
        return _merge_results(pending, final)


class _PydanticEvalAgent:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    def conversation(self) -> _PydanticEvalConversation:
        return _PydanticEvalConversation(self.runtime.conversation())

    async def run(self, message: str, **kwargs: Any) -> AgentResult:
        return await self.conversation().send(message, **kwargs)


def _comparison_model(model: str) -> Any:
    if model.startswith("openai/"):
        settings = {
            key: value
            for key, value in {
                "openai_reasoning_effort": config.HEYNYC_REASONING_EFFORT,
                "openai_service_tier": config.HEYNYC_SERVICE_TIER,
            }.items()
            if value is not None
        }
        model_type = (
            OpenAIResponsesModel if _uses_openai_responses(model) else OpenAIChatModel
        )
        return model_type(
            model.removeprefix("openai/"),
            settings=settings,
        )
    return infer_model(model.replace("/", ":", 1))


def _uses_openai_responses(model: str, *, has_tools: bool = True) -> bool:
    if not model.startswith("openai/"):
        return False
    model_info, _ = responses_api_bridge_check(
        model.removeprefix("openai/"),
        "openai",
        tools=[{}] if has_tools else [],
        reasoning_effort=config.HEYNYC_REASONING_EFFORT,
    )
    return model_info.get("mode") == "responses"


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
            "auto_confirm_confirm_star_facts_only_never_actions"
            if arm == "pydantic_ai"
            else "not_applicable"
        ),
    }


async def run_arms(
    factories: dict[str, Any],
    cases: list[Any],
    reminders: list[str],
    out_dir: Path,
    model: str,
) -> dict[str, Any]:
    receipt = {"case_ids": [case.id for case in cases], "arms": []}
    for arm, factory in factories.items():
        results = await run_all(factory, cases, reminders=reminders)
        report = await evaluate(results)
        summary = summarize_arm(arm, model, results, report)
        write_run(out_dir / arm, report, metadata=summary)
        receipt["arms"].append(summary)
    (out_dir / "comparison.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def build_factories(registry: Registry, retriever: Any, model: str) -> dict[str, Any]:
    return {
        "production": lambda: Agent(
            registry,
            model=model,
            index=retriever,
            notify_awareness=current_awareness,
            scope_gate=True,
        ),
        "pydantic_ai": lambda: _PydanticEvalAgent(
            build_runtime(
                registry,
                model=_comparison_model(model),
                answer_model_route=model,
                index=retriever,
                use_module_capabilities=True,
                current_awareness=current_awareness,
            )
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
    return parser


async def _run(args: argparse.Namespace) -> int:
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
    factories = build_factories(registry, retriever, model)
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
    )
    print(json.dumps(receipt, indent=2))
    print(f"Comparison written to {out_dir}")
    return int(any(arm["passed"] != arm["total"] for arm in receipt["arms"]))


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
