"""Eval harness — the no-hallucination gate.

Runs each module's golden `eval.yaml` cases through the agent and checks them with
deterministic assertions (expected/forbidden tools, citation kinds, abstention,
link liveness, substring expectations) plus an optional PAID API groundedness judge
(`--api-judge`; the free default is the interactive Agent reviewing the traces).
Modeled on DXA's agent_eval pattern and the SOTA RAG/ agent eval stack (Azure
Foundry / RAGAS / ALCE citation metrics).
"""
from __future__ import annotations

from .bench import BenchRow, bench_summary, render_bench, render_by_category, run_bench
from .cases import CATEGORY_NAMES, EvalCase, default_redteam_suite, load_cases, load_redteam_cases
from .checks import CheckResult, run_checks
from .judges import make_api_judge, model_family, same_family
from .redteam import run_redteam
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
    "render_by_category",
    "run_bench",
    # adversarial red-team (shared eval machinery + an independent, family-separated judge)
    "CATEGORY_NAMES",
    "default_redteam_suite",
    "load_redteam_cases",
    "make_api_judge",
    "model_family",
    "same_family",
    "run_redteam",
]
