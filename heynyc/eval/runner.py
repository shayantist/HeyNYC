"""Run eval cases through the agent and capture what happened."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ..core.agent import AgentResult
from .cases import EvalCase

NYC_TZ = ZoneInfo("America/New_York")

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


def merge_eval_results(
    pending: AgentResult,
    final: AgentResult,
    fact_confirmation_names: set[str],
) -> AgentResult:
    usage = {**pending.usage, **final.usage}
    for key in _ADDITIVE_USAGE_KEYS:
        if key in pending.usage or key in final.usage:
            usage[key] = (pending.usage.get(key) or 0) + (final.usage.get(key) or 0)
    costs = (pending.usage.get("cost_usd"), final.usage.get("cost_usd"))
    usage["cost_usd"] = (
        sum(costs) if all(isinstance(cost, (int, float)) for cost in costs) else None
    )
    usage["cost_status"] = "priced" if usage["cost_usd"] is not None else "unpriced"
    for key in ("capabilities_used", "executed_tool_calls"):
        usage[key] = list(
            dict.fromkeys(
                [
                    *(pending.usage.get(key) or ()),
                    *(final.usage.get(key) or ()),
                ]
            )
        )
    usage["model_request_ms"] = [
        *(pending.usage.get("model_request_ms") or ()),
        *(final.usage.get("model_request_ms") or ()),
    ]
    tool_calls = [*pending.tool_calls_made, *final.tool_calls_made]
    tool_calls.extend(
        name.removeprefix("confirm_").removesuffix("_facts")
        for name in final.usage.get("executed_tool_calls", ())
        if name in fact_confirmation_names
    )
    return AgentResult(
        text=final.text,
        citations={**pending.citations, **final.citations},
        tool_calls_made=tool_calls,
        iterations=pending.iterations + final.iterations,
        hit_max_iters=pending.hit_max_iters or final.hit_max_iters,
        status=final.status,
        messages=[*pending.messages, *final.messages],
        usage=usage,
    )


class PydanticEvalConversation:
    """Complete native fact review in evals; never approve an external action."""

    def __init__(self, conversation: Any):
        self.conversation = conversation

    async def send(self, message: str, **kwargs: Any) -> AgentResult:
        pending = await self.conversation.send(message, **kwargs)
        approvals = self.conversation.pending_approvals
        is_fact_confirmation = getattr(
            getattr(self.conversation, "runtime", None),
            "is_fact_confirmation",
            lambda _: False,
        )
        if (
            pending.status != "approval_required"
            or not approvals
            or not all(
                is_fact_confirmation(request["tool_name"])
                for request in approvals.values()
            )
        ):
            return pending
        self.conversation = self.conversation.runtime.conversation_from_state(
            self.conversation.dump_state()
        )
        final = await self.conversation.resume_approvals(
            {call_id: True for call_id in approvals}
        )
        return merge_eval_results(
            pending,
            final,
            {
                request["tool_name"]
                for request in approvals.values()
                if is_fact_confirmation(request["tool_name"])
            },
        )


class PydanticEvalAgent:
    """Eval-only adapter that completes structured fact review."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def conversation(self) -> PydanticEvalConversation:
        return PydanticEvalConversation(self.runtime.conversation())

    async def run(self, message: str, **kwargs: Any) -> AgentResult:
        return await self.conversation().send(message, **kwargs)


@dataclass
class CaseResult:
    case: EvalCase
    text: str = ""
    tool_calls_made: list[str] = field(default_factory=list)
    citations: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens, ...} for cost tracking
    turn_results: list[object] = field(default_factory=list)
    turn_started_at: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    error: Optional[str] = None


async def run_case(agent, case: EvalCase, reminders: Optional[list[str]] = None) -> CaseResult:
    try:
        turn_results = []
        turn_started_at = []
        if len(case.turns) > 1:
            # A conversational case plays through one Conversation so history flows exactly
            # as it does for the live surfaces; checks grade the final turn's result.
            convo = agent.conversation()
            for turn in case.turns[:-1]:
                turn_started_at.append(datetime.now(NYC_TZ).isoformat(timespec="seconds"))
                turn_results.append(await convo.send(turn, reminders=reminders))
            turn_started_at.append(datetime.now(NYC_TZ).isoformat(timespec="seconds"))
            result = await convo.send(case.turns[-1], reminders=reminders)
            turn_results.append(result)
        else:
            turn_started_at.append(datetime.now(NYC_TZ).isoformat(timespec="seconds"))
            result = await agent.run(case.query, reminders=reminders)
            turn_results.append(result)
    except Exception as exc:  # a crash is a failed case, not a crashed run
        partial = getattr(exc, "partial_result", None)
        if partial is None:
            return CaseResult(case=case, error=str(exc))
        turn_results.append(partial)
        return CaseResult(
            case=case,
            text=partial.text,
            tool_calls_made=partial.tool_calls_made,
            citations=partial.citations,
            messages=partial.messages,
            usage=partial.usage,
            turn_results=turn_results,
            turn_started_at=turn_started_at,
            diagnostics=getattr(exc, "diagnostics", {}),
            error=str(exc),
        )
    return CaseResult(
        case=case,
        text=result.text,
        tool_calls_made=result.tool_calls_made,
        citations=result.citations,
        messages=result.messages,
        usage=getattr(result, "usage", {}) or {},
        turn_results=turn_results,
        turn_started_at=turn_started_at,
    )


async def run_repeated(
    agent_factory, case: EvalCase, k: int = 3, reminders: Optional[list[str]] = None
) -> list[CaseResult]:
    """Run one case k times with a fresh agent each time (for pass^k reliability)."""
    return [await run_case(agent_factory(), case, reminders=reminders) for _ in range(k)]


async def run_all(agent_factory, cases: list[EvalCase], reminders: Optional[list[str]] = None) -> list[CaseResult]:
    """Run each case with a fresh agent (so state never leaks between cases).

    `agent_factory` is a zero-arg callable returning an Agent, keeps cases isolated.
    """
    results = []
    for case in cases:
        results.append(await run_case(agent_factory(), case, reminders=reminders))
    return results
