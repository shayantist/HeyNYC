"""Run eval cases through the agent and capture what happened."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from .cases import EvalCase

NYC_TZ = ZoneInfo("America/New_York")


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
        return CaseResult(case=case, error=str(exc))
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
