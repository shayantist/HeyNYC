"""Runtime grounding guard (agent.py): the deterministic post-generation safety hook.

After the agent produces its FINAL answer, before it reaches the user, a grounding check runs on the
cited claims. If a structured cited fact is not supported by its source, the agent gets a SPECIFIC
correction and regenerates (Tier 3, capped); if it still can't ground it, the offending claim is
stripped or the answer abstains and routes to 311 (Tier 4). A correctly-grounded answer passes through
UNCHANGED — the guard must never over-block.
"""
from __future__ import annotations

import json

import pytest

from heynyc.core.agent import GROUNDING_ABSTAIN_FALLBACK, Agent
from heynyc.core.citations import content_hash
from heynyc.core.registry import Registry
from heynyc.core.tools import Tool


def _scripted(*responses):
    """A completion fn returning the given assistant messages in order. Raises if the loop asks for
    more responses than scripted — so an unbounded retry loop fails LOUDLY instead of hanging."""
    state = {"i": 0}

    async def complete(messages, tool_schemas):
        if state["i"] >= len(responses):
            raise AssertionError(
                f"model called {state['i'] + 1} times but only {len(responses)} responses scripted "
                "— the guard is looping instead of respecting its retry cap"
            )
        resp = responses[state["i"]]
        state["i"] += 1
        return resp

    complete.calls = state
    return complete


def _assistant(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def _tool_call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _lookup_tool(snapshot, snippet=""):
    """A tool that registers a DATA citation whose complete `snapshot` is captured — so a fact absent
    from it is a CONCLUSIVE (blocking) fabrication, exactly the case the guard exists to catch."""
    async def handler(args, ctx):
        cid = ctx.citations.register(
            "https://data.cityofnewyork.us/resource/abcd-1234.json",
            snippet=snippet, kind="DATA",
            provenance={"record_id": "row-1", "field_pointer": "/",
                        "content_hash": content_hash(snapshot), "snapshot": snapshot},
        )
        return f"Found a record {{cite:{cid}}}"

    return Tool(name="lookup", description="look up a record",
                parameters={"type": "object", "properties": {}}, handler=handler)


def _agent(complete, **kw):
    tool = _lookup_tool(kw.pop("snapshot"), snippet=kw.pop("snippet", ""))
    return Agent(Registry([]), tools={"lookup": tool}, complete_fn=complete, **kw)


# --- Tier 3: catch + feedback + retry -------------------------------------------------------------

async def test_guard_catches_ungrounded_phone_then_model_fixes_it():
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("lookup", {})]),
        _assistant(content="Call New York Common Pantry at (212) 555-0100 {cite:S1}."),  # fabricated
        _assistant(content="Call New York Common Pantry at (917) 720-9700 {cite:S1}."),  # corrected
    )
    agent = _agent(complete, snapshot=snap, snippet="New York Common Pantry — Manhattan")
    result = await agent.run("food pantry near me")

    assert "(917) 720-9700" in result.text     # the corrected, grounded number ships
    assert "(212) 555-0100" not in result.text  # the fabrication never reaches the user
    assert result.iterations == 3               # tool call + rejected attempt + accepted attempt


async def test_retry_feedback_names_the_specific_offending_fact():
    # The correction fed back to the model must be SPECIFIC — it names the exact ungrounded token, so
    # the model can fix that fact rather than blindly rewrite (or worse, abstain unnecessarily).
    captured = {"messages": None}

    async def complete(messages, tool_schemas):
        # First call: tool. Second: fabrication. Third: capture what the model was told, then abstain-ish.
        n = len([m for m in messages if m.get("role") == "assistant"])
        if n == 0:
            return _assistant(tool_calls=[_tool_call("lookup", {})])
        if n == 1:
            return _assistant(content="Call them at (212) 555-0100 {cite:S1}.")
        captured["messages"] = list(messages)
        return _assistant(content="Reach New York Common Pantry at (917) 720-9700 {cite:S1}.")

    tool = _lookup_tool({"name": "New York Common Pantry", "phone": "(917) 720-9700"},
                        snippet="New York Common Pantry")
    agent = Agent(Registry([]), tools={"lookup": tool}, complete_fn=complete)
    await agent.run("food pantry near me")

    feedback = " ".join(str(m.get("content")) for m in captured["messages"] if m.get("role") == "user")
    assert "(212) 555-0100" in feedback          # names the offending value
    assert "not in" in feedback.lower() or "not supported" in feedback.lower()


# --- Tier 4: abstain after the cap, no infinite loop ----------------------------------------------

async def test_guard_abstains_after_retry_cap_and_does_not_loop():
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    bad = _assistant(content="Call them at (212) 555-0100 {cite:S1}.")
    # tool call + initial attempt + exactly guard_max_retries(2) retries = 4 model calls. A 5th call
    # (an unbounded loop) would exhaust the script and raise from _scripted.
    complete = _scripted(_assistant(tool_calls=[_tool_call("lookup", {})]), bad, bad, bad)
    agent = _agent(complete, snapshot=snap, snippet="New York Common Pantry",
                   guard_max_retries=2)
    result = await agent.run("food pantry near me", max_iters=12)

    assert "(212) 555-0100" not in result.text          # the fabrication is stripped, never ships
    # the offending claim was the whole answer (load-bearing) → abstain + route to a human/official source
    assert result.text == GROUNDING_ABSTAIN_FALLBACK
    assert "311" in result.text
    assert complete.calls["i"] == 4                       # cap respected: 1 tool + 3 terminal attempts


async def test_non_load_bearing_fact_is_stripped_answer_survives():
    # The answer has a grounded main answer PLUS a trailing sentence with a fabricated phone. After the
    # cap, the offending sentence is stripped but the grounded answer survives (no full abstention).
    snap = {"name": "New York Common Pantry", "address": "8 East 109th Street",
            "phone": "(917) 720-9700"}
    bad = _assistant(content=(
        "The nearest food pantry is New York Common Pantry at 8 East 109th Street {cite:S1}. "
        "For their hotline, call (212) 555-0100 {cite:S1}."
    ))
    complete = _scripted(_assistant(tool_calls=[_tool_call("lookup", {})]), bad, bad, bad)
    agent = _agent(complete, snapshot=snap, snippet="New York Common Pantry — Manhattan",
                   guard_max_retries=2)
    result = await agent.run("food pantry near me", max_iters=12)

    assert "(212) 555-0100" not in result.text            # fabricated hotline stripped
    assert "8 East 109th Street" in result.text           # grounded main answer preserved
    assert result.text != GROUNDING_ABSTAIN_FALLBACK


# --- No over-block: correct answers pass through unchanged -----------------------------------------

async def test_grounded_answer_passes_through_unchanged_no_retry():
    snap = {"name": "New York Common Pantry", "address": "8 East 109th Street",
            "phone": "(917) 720-9700"}
    final = ("New York Common Pantry is at 8 East 109th Street {cite:S1}. "
             "Call (917) 720-9700 {cite:S1}.")
    complete = _scripted(_assistant(tool_calls=[_tool_call("lookup", {})]),
                         _assistant(content=final))
    agent = _agent(complete, snapshot=snap, snippet="New York Common Pantry — Manhattan")
    result = await agent.run("food pantry near me")

    assert result.text == final        # byte-for-byte unchanged
    assert result.iterations == 2      # no guard retry iteration was spent
    assert complete.calls["i"] == 2    # exactly one terminal completion, no regeneration


async def test_answer_with_no_citations_is_not_guarded():
    complete = _scripted(_assistant(content="I don't have that info — try 311 or nyc.gov."))
    agent = Agent(Registry([]), tools={}, complete_fn=complete)
    result = await agent.run("where's the nearest cooling center?")
    assert result.text == "I don't have that info — try 311 or nyc.gov."
    assert complete.calls["i"] == 1


async def test_soft_snippet_mismatch_does_not_trigger_guard():
    # A number absent from a TRUNCATED web/doc snippet is a SOFT mismatch (it might be elsewhere on the
    # page) — the guard must NOT block or rewrite it, or it would over-abstain on real answers.
    async def handler(args, ctx):
        cid = ctx.citations.register("https://www.nyc.gov/notify", kind="WEB",
                                     snippet="A heat advisory is in effect today.",
                                     title="Notify NYC")
        return f"ok {{cite:{cid}}}"

    tool = Tool(name="weather", description="weather", parameters={"type": "object", "properties": {}},
                handler=handler)
    final = "There's a heat advisory with highs near 95°F {cite:S1}."
    complete = _scripted(_assistant(tool_calls=[_tool_call("weather", {})]),
                         _assistant(content=final))
    agent = Agent(Registry([]), tools={"weather": tool}, complete_fn=complete)
    result = await agent.run("what's the weather advisory?")
    assert result.text == final        # unchanged; the soft mismatch never fires the guard
    assert complete.calls["i"] == 2


async def test_guard_can_be_disabled():
    # Escape hatch: with the guard off, even a fabricated fact ships (so the flag is observable).
    snap = {"name": "New York Common Pantry", "phone": "(917) 720-9700"}
    complete = _scripted(_assistant(tool_calls=[_tool_call("lookup", {})]),
                         _assistant(content="Call (212) 555-0100 {cite:S1}."))
    agent = _agent(complete, snapshot=snap, snippet="NYCP", guard_grounding=False)
    result = await agent.run("food pantry near me")
    assert "(212) 555-0100" in result.text
    assert complete.calls["i"] == 2
