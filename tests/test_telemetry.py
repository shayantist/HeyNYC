from __future__ import annotations

from pathlib import Path

from heynyc.core import telemetry


def test_cost_usd_is_explicit_for_unknown_model():
    assert telemetry.cost_usd("definitely-not-a-real-model", 100, 50) is None


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


def test_record_turn_preserves_explicit_unpriceable_cost(tmp_path: Path):
    rec = telemetry.record_turn(
        tmp_path / "telemetry.jsonl", session_id="s1", model="unknown/model",
        usage={"input_tokens": 10, "output_tokens": 2, "cost_usd": None,
               "cost_status": "unpriced", "scope_model": "unknown/scope"},
        n_tool_calls=0, tool_names=[], status="success",
    )

    assert rec["cost_usd"] is None
    assert rec["cost_status"] == "unpriced"
    assert rec["scope_model"] == "unknown/scope"


def test_record_turn_preserves_memory_compaction_accounting(tmp_path: Path):
    rec = telemetry.record_turn(
        tmp_path / "telemetry.jsonl", session_id="s1", model="answer/model",
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "memory_compactions": 1,
            "memory_model": "openai/gpt-5.4-nano",
            "memory_input_tokens": 30,
            "memory_output_tokens": 5,
            "memory_cost_usd": 0.001,
            "memory_time_ms": 50.0,
            "memory_pre_tokens": 1_500,
            "memory_post_tokens": 700,
        },
        n_tool_calls=0, tool_names=[], status="success",
    )

    assert rec["memory_compactions"] == 1
    assert rec["memory_model"] == "openai/gpt-5.4-nano"
    assert rec["memory_pre_tokens"] == 1_500
    assert rec["memory_post_tokens"] == 700


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
    assert s["unpriced_turns"] == 0


def test_summarize_accepts_legacy_records_without_latency_breakdown():
    summary = telemetry.summarize([{
        "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
        "latency_ms": 10.0, "tool_names": [], "status": "success",
    }])

    assert summary["model_time_ms"] == 0.0
    assert summary["tool_time_ms"] == 0.0
    assert summary["scope_time_ms"] == 0.0
    assert summary["orchestration_time_ms"] == 0.0
    assert summary["n_model_calls"] == 0
    assert summary["n_tool_calls"] == 0
    assert summary["iterations"] == 0
    assert summary["memory_compactions"] == 0
    assert summary["memory_time_ms"] == 0.0


def test_record_turn_persists_checklist_and_summarize_aggregates_it(tmp_path):
    """The scope checklist is only useful if it lands in telemetry and the stats command can
    show its distribution, alongside outcome rates, so routing drift is visible passively."""
    from heynyc.core import telemetry

    path = tmp_path / "t.jsonl"
    for modules, outcome in (
        (["events", "advisories"], "answered"),
        (["events"], "answered"),
        ([], "redirected"),
    ):
        telemetry.record_turn(
            path, session_id="u1", model="m", n_tool_calls=0, tool_names=[],
            status="success",
            usage={"scope_modules": modules, "scope_situations": [], "cost_usd": 0.0},
            extra={"outcome": outcome},
        )

    records = telemetry.load(path)
    assert records[0]["scope_modules"] == ["events", "advisories"]

    summary = telemetry.summarize(records)
    assert summary["outcome_mix"] == {"answered": 2, "redirected": 1}
    assert summary["scope_module_mix"] == {"events": 2, "advisories": 1}
    assert summary["scope_situation_mix"] == {}
