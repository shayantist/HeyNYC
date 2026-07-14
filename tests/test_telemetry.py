from __future__ import annotations

from pathlib import Path

from heynyc.core import telemetry


def test_cost_usd_is_zero_for_unknown_model():
    # never crash on an unpriced/mock model — telemetry must not break a turn
    assert telemetry.cost_usd("definitely-not-a-real-model", 100, 50) == 0.0


def test_record_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    rec = telemetry.record_turn(
        path, session_id="s1", model="m",
        usage={"input_tokens": 10, "output_tokens": 5, "latency_ms": 120.0,
               "model_time_ms": 90.0, "tool_time_ms": 20.0, "n_model_calls": 2,
               "iterations": 2},
        n_tool_calls=1, tool_names=["benefits_search"], status="success",
    )
    assert rec["input_tokens"] == 10
    assert rec["session_id"] == "s1"
    assert rec["model_time_ms"] == 90.0
    assert rec["orchestration_time_ms"] == 0.0
    assert rec["n_model_calls"] == 2
    loaded = telemetry.load(path)
    assert len(loaded) == 1 and loaded[0]["tool_names"] == ["benefits_search"]


def test_summarize_aggregates():
    records = [
        {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50, "latency_ms": 100.0,
         "tool_names": ["benefits_search"], "status": "success"},
        {"cost_usd": 0.03, "input_tokens": 200, "output_tokens": 80, "latency_ms": 300.0,
         "tool_names": ["nearest", "benefits_search"], "status": "error"},
    ]
    s = telemetry.summarize(records)
    assert s["turns"] == 2
    assert abs(s["total_cost_usd"] - 0.04) < 1e-9
    assert s["input_tokens"] == 300
    assert s["tool_mix"]["benefits_search"] == 2
    assert s["error_rate"] == 0.5
    assert 100.0 <= s["latency_p50_ms"] <= 300.0  # numpy.percentile (linear interp) of [100, 300]


def test_summarize_empty():
    s = telemetry.summarize([])
    assert s["turns"] == 0 and s["total_cost_usd"] == 0.0


def test_summarize_accepts_legacy_records_without_latency_breakdown():
    summary = telemetry.summarize([{
        "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
        "latency_ms": 10.0, "tool_names": [], "status": "success",
    }])

    assert summary["model_time_ms"] == 0.0
    assert summary["tool_time_ms"] == 0.0
    assert summary["orchestration_time_ms"] == 0.0
    assert summary["n_model_calls"] == 0
    assert summary["n_tool_calls"] == 0
    assert summary["iterations"] == 0
