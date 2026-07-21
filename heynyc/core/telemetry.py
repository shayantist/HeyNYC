"""Per-turn cost/usage telemetry, internal operational metrics, $0, no paid SaaS.

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

import numpy as np  # already a dep (core/index/store.py), reuse for percentiles


def priced_cost_usd(
    model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
) -> float | None:
    """USD cost from LiteLLM, or None when the model cannot be priced.

    Cache-aware: `cached_input_tokens` (prompt_tokens_details.cached_tokens) is billed at the
    model's cache-read rate via LiteLLM's own `cost_per_token(cache_read_input_tokens=...)`, which
    prices `(prompt_tokens - cache_read) * input_cost + cache_read * cache_read_input_token_cost`.
    `input_tokens` is the TOTAL prompt count (cached included), matching LiteLLM's contract. We
    clamp the cached count to input (a provider can never cache-read more than it was sent) so the
    money math can never go negative."""
    try:
        import litellm

        cached = max(0, min(int(cached_input_tokens), int(input_tokens)))
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=int(input_tokens), completion_tokens=int(output_tokens),
            cache_read_input_tokens=cached,
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return None


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
) -> float | None:
    """USD cost, explicitly None when LiteLLM cannot price the model."""
    return priced_cost_usd(model, input_tokens, output_tokens, cached_input_tokens)


def default_path(data_dir: Path) -> Path:
    """The telemetry log path under the app's data dir (injected, no domain config)."""
    return Path(data_dir) / "telemetry.jsonl"


def record_turn(
    path: Path, *, session_id: str, model: str, usage: dict,
    n_tool_calls: int, tool_names: list[str], status: str, extra: dict | None = None,
) -> dict:
    """Append one per-turn telemetry record (JSONL) and return it.

    `extra` merges channel-level fields (channel/outcome/n_citations) into the record
    for the messaging on-ramp; `summarize()` ignores unknown keys."""
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
    turn_cost = usage.get("cost_usd") if "cost_usd" in usage else priced_cost_usd(
        model, input_tokens, output_tokens, cached_input_tokens
    )
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": turn_cost,
        "cost_status": usage.get("cost_status") or ("priced" if turn_cost is not None else "unpriced"),
        "latency_ms": float(usage.get("latency_ms", 0.0) or 0.0),
        "model_time_ms": float(usage.get("model_time_ms", 0.0) or 0.0),
        "tool_time_ms": float(usage.get("tool_time_ms", 0.0) or 0.0),
        "orchestration_time_ms": float(usage.get("orchestration_time_ms", 0.0) or 0.0),
        "n_model_calls": int(usage.get("n_model_calls", 0) or 0),
        "n_tool_calls": n_tool_calls,
        "iterations": int(usage.get("iterations", 0) or 0),
        "tool_names": list(tool_names),
        "status": status,
    }
    for key in (
        "cached_input_tokens", "scope_cached_input_tokens",
        "answer_input_tokens", "answer_output_tokens", "scope_input_tokens",
        "scope_output_tokens", "scope_model", "scope_cost_usd", "scope_time_ms",
        "memory_compactions", "memory_model", "memory_input_tokens",
        "memory_output_tokens", "memory_cost_usd", "memory_time_ms",
        "memory_pre_tokens", "memory_post_tokens",
        "scope_modules", "scope_situations",
    ):
        if key in usage:
            record[key] = usage[key]
    if extra:
        record.update(extra)
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
                "unpriced_turns": 0,
                "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
                "scope_cached_input_tokens": 0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0, "model_time_ms": 0.0, "tool_time_ms": 0.0,
                "scope_time_ms": 0.0, "orchestration_time_ms": 0.0,
                "memory_compactions": 0, "memory_time_ms": 0.0,
                "n_model_calls": 0, "n_tool_calls": 0, "iterations": 0,
                "tool_mix": {}, "error_rate": 0.0,
                "outcome_mix": {}, "scope_module_mix": {}, "scope_situation_mix": {}}
    turns = len(records)
    total_cost = sum(float(r["cost_usd"]) for r in records if r.get("cost_usd") is not None)
    unpriced_turns = sum(1 for r in records if r.get("cost_usd") is None)
    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    tool_mix = Counter(t for r in records for t in r.get("tool_names", []))
    errors = sum(1 for r in records if r.get("status") not in ("success", None))
    return {
        "turns": turns,
        "total_cost_usd": total_cost,
        "cost_per_turn_usd": total_cost / turns,
        "unpriced_turns": unpriced_turns,
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in records),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in records),
        "cached_input_tokens": sum(int(r.get("cached_input_tokens", 0) or 0) for r in records),
        "scope_cached_input_tokens": sum(
            int(r.get("scope_cached_input_tokens", 0) or 0) for r in records
        ),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "model_time_ms": sum(float(r.get("model_time_ms", 0.0) or 0.0) for r in records),
        "tool_time_ms": sum(float(r.get("tool_time_ms", 0.0) or 0.0) for r in records),
        "scope_time_ms": sum(float(r.get("scope_time_ms", 0.0) or 0.0) for r in records),
        "memory_compactions": sum(
            int(r.get("memory_compactions", 0) or 0) for r in records
        ),
        "memory_time_ms": sum(float(r.get("memory_time_ms", 0.0) or 0.0) for r in records),
        "orchestration_time_ms": sum(
            float(r.get("orchestration_time_ms", 0.0) or 0.0) for r in records
        ),
        "n_model_calls": sum(int(r.get("n_model_calls", 0) or 0) for r in records),
        "n_tool_calls": sum(int(r.get("n_tool_calls", 0) or 0) for r in records),
        "iterations": sum(int(r.get("iterations", 0) or 0) for r in records),
        "tool_mix": dict(tool_mix),
        "error_rate": errors / turns,
        # Passive routing-drift visibility: what the checklist marked and how turns resolved.
        "outcome_mix": dict(Counter(
            r["outcome"] for r in records if r.get("outcome")
        )),
        "scope_module_mix": dict(Counter(
            m for r in records for m in r.get("scope_modules", []) or []
        )),
        "scope_situation_mix": dict(Counter(
            s for r in records for s in r.get("scope_situations", []) or []
        )),
    }
