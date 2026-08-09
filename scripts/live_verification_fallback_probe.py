#!/usr/bin/env python3
"""Run the F179 verification fallback through the live runtime with an eval-only fault."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from heynyc.core import config
from heynyc.core.localization import localize
from heynyc.core.pydantic_runtime import configured_model
from heynyc.core.pydantic_runtime.runtime import VERIFICATION_ABSTAIN_FALLBACK
from heynyc.core.registry import Registry
from heynyc.eval import evaluate, load_cases, run_case, write_run
from heynyc.eval.bench import _candidate_cost, build_eval_agent
from heynyc.eval.faults import VerificationFallbackProbeModel, verified_fallback_probe

_CASE_ID = "snap_where_to_apply_in_person__es"


async def run_probe(out_dir: Path) -> None:
    registry = Registry.discover(
        config.MODULES_DIR,
        config.BASE_ALLOWLIST,
        config.NEWS_ALLOWLIST,
    )
    case = next(case for case in load_cases(registry) if case.id == _CASE_ID)
    model_name = config.HEYNYC_MODEL
    agent = build_eval_agent(registry, model_name, retriever=None)
    fault_model = VerificationFallbackProbeModel(configured_model(model_name))

    with agent.runtime._agent.override(model=fault_model):
        result = await run_case(agent, case)

    expected = localize(VERIFICATION_ABSTAIN_FALLBACK, "es")
    rejections = result.diagnostics.get("validation_rejections") or []
    report = await evaluate([result])
    cost, input_tokens, output_tokens = _candidate_cost(model_name, [result])
    write_run(
        out_dir,
        report,
        metadata={
            "probe": "F179 verification fallback",
            "model": model_name,
            "case_ids": [_CASE_ID],
            "candidate_cost_usd": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "n_model_calls": result.usage.get("n_model_calls", 0),
            "n_tool_calls": result.usage.get("n_tool_calls", 0),
        },
    )
    if not verified_fallback_probe(result, expected):
        raise RuntimeError("F179 probe did not reach the verified Spanish fallback boundary")

    cost_text = "unpriced" if cost is None else f"${cost:.8f}"
    print(
        f"F179 fallback observed: {len(rejections)} validation rejections, "
        f"{result.usage.get('n_model_calls', 0)} model calls, cost {cost_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_probe(args.out))


if __name__ == "__main__":
    main()
