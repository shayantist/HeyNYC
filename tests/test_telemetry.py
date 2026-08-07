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


def test_record_turn_and_summary_expose_safety_screen_overhead(tmp_path: Path):
    rec = telemetry.record_turn(
        tmp_path / "telemetry.jsonl",
        session_id="s1",
        model="answer/model",
        usage={
            "input_tokens": 110,
            "output_tokens": 22,
            "safety_model": "openai/gpt-5.4-mini",
            "safety_input_tokens": 10,
            "safety_output_tokens": 2,
            "safety_cached_input_tokens": 4,
            "safety_cost_usd": 0.0005,
            "safety_time_ms": 12.0,
        },
        n_tool_calls=0,
        tool_names=[],
        status="success",
    )

    assert rec["safety_model"] == "openai/gpt-5.4-mini"
    assert rec["safety_cost_usd"] == 0.0005
    summary = telemetry.summarize([rec])
    assert summary["safety_input_tokens"] == 10
    assert summary["safety_output_tokens"] == 2
    assert summary["safety_cost_usd"] == 0.0005
    assert summary["safety_time_ms"] == 12.0


def test_safety_screen_failure_is_persisted_and_counts_as_an_operational_error(
    tmp_path: Path,
):
    rec = telemetry.record_turn(
        tmp_path / "telemetry.jsonl",
        session_id="s1",
        model="answer/model",
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "safety_error": "TimeoutError",
        },
        n_tool_calls=0,
        tool_names=[],
        status="success",
    )

    assert rec["status"] == "success"
    assert rec["safety_error"] == "TimeoutError"
    assert telemetry.summarize([rec])["error_rate"] == 1.0


def test_record_turn_passes_through_cached_input_tokens(tmp_path: Path):
    # The agent already computes prompt-cache reads (usage["cached_input_tokens"]); telemetry must
    # not drop them, so `heynyc stats` can show cache effectiveness.
    rec = telemetry.record_turn(
        tmp_path / "telemetry.jsonl", session_id="s1", model="answer/model",
        usage={"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 64},
        n_tool_calls=0, tool_names=[], status="success",
    )
    assert rec["cached_input_tokens"] == 64


def test_summarize_sums_cached_input_tokens():
    records = [
        {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50, "latency_ms": 100.0,
         "cached_input_tokens": 40, "tool_names": [], "status": "success"},
        {"cost_usd": 0.02, "input_tokens": 200, "output_tokens": 80, "latency_ms": 200.0,
         "cached_input_tokens": 25, "tool_names": [], "status": "success"},
    ]
    s = telemetry.summarize(records)
    assert s["cached_input_tokens"] == 65
    # Legacy records without the field aggregate to zero, never KeyError.
    assert telemetry.summarize([])["cached_input_tokens"] == 0


def test_summarize_sums_scope_cached_input_tokens_separately():
    # Cache-layout fix: the scope call's cache read is captured into telemetry so its cache rate is
    # visible in `heynyc stats` alongside the answer call's, not silently folded away.
    records = [
        {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50, "latency_ms": 100.0,
         "cached_input_tokens": 90, "scope_cached_input_tokens": 60,
         "tool_names": [], "status": "success"},
        {"cost_usd": 0.02, "input_tokens": 200, "output_tokens": 80, "latency_ms": 200.0,
         "cached_input_tokens": 30, "scope_cached_input_tokens": 20,
         "tool_names": [], "status": "success"},
    ]
    s = telemetry.summarize(records)
    assert s["scope_cached_input_tokens"] == 80
    assert telemetry.summarize([])["scope_cached_input_tokens"] == 0


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


_CACHE_TEST_MODEL = "heynyc-cachetest/model"


def _register_cache_test_model():
    import litellm

    litellm.register_model({
        _CACHE_TEST_MODEL: {
            "input_cost_per_token": 1e-05,          # $10 / M
            "cache_read_input_token_cost": 1e-06,   # $1 / M  (10x cheaper, like the real models)
            "output_cost_per_token": 3e-05,         # $30 / M
            "litellm_provider": "openai", "mode": "chat",
        }
    })


def test_priced_cost_prices_cached_input_at_reduced_rate():
    # Cache-aware accounting: cached prompt tokens bill at the model's cache-read rate via litellm's
    # own cost_per_token(cache_read_input_tokens=...), not at the full input rate.
    _register_cache_test_model()
    full = telemetry.priced_cost_usd(_CACHE_TEST_MODEL, 1000, 200)
    cached = telemetry.priced_cost_usd(_CACHE_TEST_MODEL, 1000, 200, cached_input_tokens=600)
    assert abs(full - 0.016) < 1e-12                       # 1000*1e-5 + 200*3e-5
    assert abs(cached - 0.0106) < 1e-12                    # 400*1e-5 + 600*1e-6 + 200*3e-5
    assert cached < full                                   # caching must lower the bill


def test_priced_cost_clamps_cached_at_or_below_input():
    # Defensive money math: a provider reporting more cached tokens than total input never yields a
    # negative bill (would happen if litellm subtracted cached from a smaller prompt count).
    _register_cache_test_model()
    cost = telemetry.priced_cost_usd(_CACHE_TEST_MODEL, 5, 1, cached_input_tokens=8)
    assert cost is not None and cost >= 0.0


def test_record_turn_fallback_cost_is_cache_aware(tmp_path):
    # When the agent does not pre-compute cost_usd, record_turn's own pricing must still be
    # cache-aware, using the record's cached_input_tokens.
    _register_cache_test_model()
    rec = telemetry.record_turn(
        tmp_path / "t.jsonl", session_id="s", model=_CACHE_TEST_MODEL,
        usage={"input_tokens": 1000, "output_tokens": 200, "cached_input_tokens": 600},
        n_tool_calls=0, tool_names=[], status="success",
    )
    assert abs(rec["cost_usd"] - 0.0106) < 1e-12
