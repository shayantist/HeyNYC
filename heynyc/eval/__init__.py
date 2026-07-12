"""Eval harness — the no-hallucination gate.

Runs each module's golden `eval.yaml` cases through the agent and checks them with
deterministic assertions (expected/forbidden tools, citation kinds, abstention,
link liveness, substring expectations) plus an optional PAID API groundedness judge
(`--api-judge`; the free default is the interactive Agent reviewing the traces).
Modeled on DXA's agent_eval pattern and the SOTA RAG/ agent eval stack (Azure
Foundry / RAGAS / ALCE citation metrics).
"""
from __future__ import annotations

from .bench import BenchRow, bench_summary, render_bench, run_bench
from .cases import EvalCase, load_cases
from .checks import CheckResult, run_checks
from .redteam import (
    CATEGORIES,
    RedTeamCase,
    RedTeamCaseResult,
    RedTeamGrader,
    RedTeamReport,
    RedTeamVerdict,
    load_suite,
    model_family,
    reconcile,
    run_redteam,
    same_family,
)
from .report import GateReport, evaluate, write_run
from .runner import CaseResult, run_all, run_case, run_repeated
from .trace import Trace, build_trace

__all__ = [
    "EvalCase",
    "load_cases",
    "CheckResult",
    "run_checks",
    "CaseResult",
    "run_case",
    "run_all",
    "run_repeated",
    "GateReport",
    "evaluate",
    "write_run",
    "Trace",
    "build_trace",
    "BenchRow",
    "bench_summary",
    "render_bench",
    "run_bench",
    # red-team harness (independent-grader adversarial suite)
    "CATEGORIES",
    "RedTeamCase",
    "RedTeamCaseResult",
    "RedTeamGrader",
    "RedTeamReport",
    "RedTeamVerdict",
    "load_suite",
    "model_family",
    "reconcile",
    "run_redteam",
    "same_family",
]
