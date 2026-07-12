"""Spend-cap guard (security-audit F2b / OWASP LLM10 Unbounded Consumption).

Offline: cost is injected, so nothing here prices a real model or spends a cent. The guard
reuses core.telemetry.cost_usd for real pricing (asserted by identity, not by calling it).
"""
from __future__ import annotations

import pytest

from heynyc.core import spend, telemetry
from heynyc.core.registry import Registry


# --- Unit: the SpendGuard itself ------------------------------------------------------------------

def test_default_cost_fn_is_the_telemetry_pricing_path():
    # "computes cost via the existing telemetry path": no reinvented price table.
    assert spend.SpendGuard(1.0)._cost_fn is telemetry.cost_usd


def test_disabled_when_no_cap_and_is_a_noop():
    # Default OFF: no cap (None) or a non-positive cap disables the guard, and recording a
    # huge cost neither accumulates nor ever halts, so agent behavior is unchanged.
    for cap in (None, 0, 0.0, -5):
        g = spend.SpendGuard(cap, cost_fn=lambda m, i, o: 1_000_000.0)
        assert not g.enabled
        assert g.record("anthropic/claude-sonnet-4-6", 10, 10) == 0.0
        assert g.spent_usd == 0.0
        assert g.halt_reason() is None


def test_accumulates_injected_cost_and_halts_when_cap_met_or_exceeded():
    g = spend.SpendGuard(0.10, cost_fn=lambda m, i, o: 0.04)
    assert g.record("m", 1, 1) == 0.04
    assert g.halt_reason() is None                 # 0.04 < 0.10, keep going
    g.record("m", 1, 1)                            # 0.08 < 0.10
    assert g.halt_reason() is None
    g.record("m", 1, 1)                            # 0.12 >= 0.10 -> halt
    reason = g.halt_reason()
    assert reason is not None and "spend cap reached" in reason


def test_halts_exactly_at_the_cap_boundary():
    g = spend.SpendGuard(1.0, cost_fn=lambda m, i, o: 1.0)
    g.record("m", 1, 1)                            # spent == cap: "meets or exceeds"
    assert g.halt_reason() is not None


def test_failsafe_uncomputable_cost_under_active_cap_halts():
    # If cost cannot be computed while a cap is active, do NOT count it as $0 (that would
    # silently disable the cap). The guard latches into a halt state instead.
    def boom(model, i, o):
        raise RuntimeError("pricing unavailable")

    g = spend.SpendGuard(5.0, cost_fn=boom)
    assert g.record("m", 1, 1) == 0.0             # swallowed, not crashing the turn
    reason = g.halt_reason()
    assert reason is not None and "could not verify" in reason


def test_failsafe_does_not_fire_when_cap_is_disabled():
    # With no cap, an uncomputable cost changes nothing: still a no-op, still never halts.
    def boom(model, i, o):
        raise RuntimeError("pricing unavailable")

    g = spend.SpendGuard(None, cost_fn=boom)
    assert g.record("m", 1, 1) == 0.0
    assert g.halt_reason() is None


# --- Agent wiring: the turn-boundary hook ---------------------------------------------------------

def _tool(name="noop"):
    from heynyc.core.tools import Tool

    async def handler(args, ctx):
        return "ok"

    return Tool(name=name, description="x", parameters={"type": "object", "properties": {}}, handler=handler)


async def test_agent_halts_further_model_calls_when_cap_exceeded():
    from heynyc.core import events
    from heynyc.core.agent import Agent, SPEND_CAPPED_FALLBACK

    calls = {"n": 0}

    async def fake_stream(messages, tool_schemas):
        calls["n"] += 1
        yield {"type": "usage", "input_tokens": 100, "output_tokens": 100}
        # Always make a tool call so the loop wants a SECOND model call (which the cap must block).
        yield {"type": "message", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}]}}

    agent = Agent(Registry([]), tools={"noop": _tool()}, stream_fn=fake_stream)
    # Inject a fake cost: each model call "costs" $1, well past the $0.50 cap.
    agent._spend = spend.SpendGuard(0.50, cost_fn=lambda m, i, o: 1.0)

    seen = [e async for e in agent.stream("go")]

    assert calls["n"] == 1                          # the SECOND model call never happened
    done = next(e for e in seen if isinstance(e, events.Done))
    assert done.status == "max_budget"
    err = next(e for e in seen if isinstance(e, events.ErrorEvent))
    assert err.scope == "spend"
    result = done.result
    assert result.status == "max_budget"
    assert result.text == SPEND_CAPPED_FALLBACK


async def test_agent_runs_normally_when_cap_disabled_even_with_usage():
    # spend_cap unset (default): the guard is a no-op, so a turn that reports token usage still
    # completes normally. Behavior is unchanged without a cap.
    from heynyc.core import config, events
    from heynyc.core.agent import Agent

    async def fake_stream(messages, tool_schemas):
        yield {"type": "text", "text": "Hello"}
        yield {"type": "usage", "input_tokens": 9999, "output_tokens": 9999}
        yield {"type": "message", "message": {"role": "assistant", "content": "Hello", "tool_calls": None}}

    agent = Agent(Registry([]), tools={}, stream_fn=fake_stream, spend_cap=None)
    assert not agent._spend.enabled
    done = [e async for e in agent.stream("hi") if isinstance(e, events.Done)][0]
    assert done.status == "success"
    assert done.result.text == "Hello"


def test_agent_reads_spend_cap_from_config_by_default(monkeypatch):
    from heynyc.core import agent as agent_mod
    from heynyc.core import config

    monkeypatch.setattr(config, "HEYNYC_SPEND_CAP", 2.5, raising=False)
    a = agent_mod.Agent(Registry([]), tools={})
    assert a._spend.enabled and a._spend.cap_usd == 2.5

    monkeypatch.setattr(config, "HEYNYC_SPEND_CAP", None, raising=False)
    b = agent_mod.Agent(Registry([]), tools={})
    assert not b._spend.enabled


def test_explicit_spend_cap_overrides_config(monkeypatch):
    from heynyc.core import agent as agent_mod
    from heynyc.core import config

    monkeypatch.setattr(config, "HEYNYC_SPEND_CAP", None, raising=False)
    a = agent_mod.Agent(Registry([]), tools={}, spend_cap=0.25)
    assert a._spend.enabled and a._spend.cap_usd == 0.25
