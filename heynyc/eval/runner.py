"""Run eval cases through the agent and capture what happened."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cases import EvalCase


@dataclass
class CaseResult:
    case: EvalCase
    text: str = ""
    tool_calls_made: list[str] = field(default_factory=list)
    citations: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens, ...} for cost tracking
    error: Optional[str] = None


async def run_case(agent, case: EvalCase, reminders: Optional[list[str]] = None) -> CaseResult:
    try:
        result = await agent.run(case.query, reminders=reminders)
    except Exception as exc:  # a crash is a failed case, not a crashed run
        return CaseResult(case=case, error=str(exc))
    return CaseResult(
        case=case,
        text=result.text,
        tool_calls_made=result.tool_calls_made,
        citations=result.citations,
        messages=result.messages,
        usage=getattr(result, "usage", {}) or {},
    )


async def run_repeated(
    agent_factory, case: EvalCase, k: int = 3, reminders: Optional[list[str]] = None
) -> list[CaseResult]:
    """Run one case k times with a fresh agent each time (for pass^k reliability)."""
    return [await run_case(agent_factory(), case, reminders=reminders) for _ in range(k)]


async def run_all(agent_factory, cases: list[EvalCase], reminders: Optional[list[str]] = None) -> list[CaseResult]:
    """Run each case with a fresh agent (so state never leaks between cases).

    `agent_factory` is a zero-arg callable returning an Agent — keeps cases isolated.
    """
    results = []
    for case in cases:
        results.append(await run_case(agent_factory(), case, reminders=reminders))
    return results
