"""Per-turn cost/usage telemetry — internal operational metrics, $0, no paid SaaS.

Tokens come from LiteLLM (the agent attaches them to AgentResult.usage); cost is
LiteLLM's own `cost_per_token`. Records are appended as JSONL and aggregated by
`heynyc stats`. This is operational data (cost/latency/tokens/tools), NOT per-user
behavioral profiling (spec §13).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # already a dep (core/index/store.py) — reuse for percentiles

from . import config


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a call via LiteLLM's pricing; 0.0 for unknown/mock models (never raises)."""
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=int(input_tokens), completion_tokens=int(output_tokens)
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return 0.0


def default_path() -> Path:
    return config.HEYNYC_DATA_DIR / "telemetry.jsonl"


def record_turn(
    path: Path, *, session_id: str, model: str, usage: dict,
    n_tool_calls: int, tool_names: list[str], status: str,
) -> dict:
    """Append one per-turn telemetry record (JSONL) and return it."""
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd(model, input_tokens, output_tokens),
        "latency_ms": float(usage.get("latency_ms", 0.0) or 0.0),
        "n_tool_calls": n_tool_calls,
        "tool_names": list(tool_names),
        "status": status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-turn records into the `heynyc stats` dashboard numbers."""
    if not records:
        return {"turns": 0, "total_cost_usd": 0.0, "cost_per_turn_usd": 0.0,
                "input_tokens": 0, "output_tokens": 0, "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0, "tool_mix": {}, "error_rate": 0.0}
    turns = len(records)
    total_cost = sum(float(r.get("cost_usd", 0.0)) for r in records)
    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    tool_mix = Counter(t for r in records for t in r.get("tool_names", []))
    errors = sum(1 for r in records if r.get("status") not in ("success", None))
    return {
        "turns": turns,
        "total_cost_usd": total_cost,
        "cost_per_turn_usd": total_cost / turns,
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in records),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in records),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "tool_mix": dict(tool_mix),
        "error_rate": errors / turns,
    }
