"""Eval harness — the no-hallucination gate.

Runs each module's golden `eval.yaml` cases through the agent and checks them with
deterministic assertions (expected/forbidden tools, citation kinds, abstention,
link liveness, substring expectations) plus an optional LLM judge (groundedness,
abstention appropriateness). Modeled on DXA's agent_eval pattern and the SOTA RAG/
agent eval stack (Azure Foundry / RAGAS / ALCE citation metrics).
"""
from __future__ import annotations

from .cases import EvalCase, load_cases
from .checks import CheckResult, run_checks
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
]
