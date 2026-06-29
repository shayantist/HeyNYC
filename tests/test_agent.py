from __future__ import annotations

import json

import pytest

from heynyc.core.agent import Agent
from heynyc.core.registry import Registry
from heynyc.core.tools import Tool, ToolContext


def _scripted(*responses):
    """Build a completion fn that returns the given assistant messages in order."""
    calls = {"i": 0}

    async def complete(messages, tool_schemas):
        resp = responses[calls["i"]]
        calls["i"] += 1
        return resp

    return complete


def _assistant(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def _tool_call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def empty_registry():
    return Registry([])


async def test_abstains_with_no_tools(empty_registry):
    complete = _scripted(_assistant(content="I don't have that info — try nyc.gov."))
    agent = Agent(empty_registry, tools={}, complete_fn=complete)
    result = await agent.run("where's the nearest cooling center?")
    assert "nyc.gov" in result.text
    assert result.iterations == 1
    assert result.tool_calls_made == []
    assert not result.hit_max_iters


async def test_tool_call_then_final_answer(empty_registry):
    async def nearest(args, ctx: ToolContext):
        cid = ctx.citations.register(
            "https://data.cityofnewyork.us/resource/h2bn-gu9k.json",
            snippet="Cooling site at 120 Broadway",
            kind="DATA",
        )
        return f"Nearest: 120 Broadway (0.2 mi) {{cite:{cid}}}"

    tool = Tool(
        name="nearest",
        description="find nearest",
        parameters={"type": "object", "properties": {"category": {"type": "string"}}},
        handler=nearest,
    )
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("nearest", {"category": "cooling_center"})]),
        _assistant(content="The nearest cooling center is 120 Broadway, 0.2 mi away {cite:S1}."),
    )
    agent = Agent(empty_registry, tools={"nearest": tool}, complete_fn=complete)
    result = await agent.run("nearest cooling center to me?")

    assert result.tool_calls_made == ["nearest"]
    assert result.iterations == 2
    assert "S1" in result.citations
    assert result.citations["S1"]["kind"] == "DATA"
    assert "120 Broadway" in result.text


async def test_unknown_tool_surfaces_error_not_crash(empty_registry):
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("ghost", {})]),
        _assistant(content="Sorry, something went wrong."),
    )
    agent = Agent(empty_registry, tools={}, complete_fn=complete)
    result = await agent.run("do something")
    # tool result with the error is fed back; loop continues to a final answer
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]
    assert result.iterations == 2


async def test_handler_exception_surfaced(empty_registry):
    async def boom(args, ctx):
        raise RuntimeError("socrata down")

    tool = Tool(name="boom", description="x", parameters={"type": "object", "properties": {}}, handler=boom)
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("boom", {})]),
        _assistant(content="I couldn't reach the data source right now."),
    )
    agent = Agent(empty_registry, tools={"boom": tool}, complete_fn=complete)
    result = await agent.run("go")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert "socrata down" in tool_msgs[0]["content"]


async def test_conversation_threads_history(empty_registry):
    seen_messages = []

    async def recorder(messages, tool_schemas):
        seen_messages.append([m for m in messages if m["role"] in ("user", "assistant")])
        return _assistant(content=f"answer {len(seen_messages)}")

    agent = Agent(empty_registry, tools={}, complete_fn=recorder)
    convo = agent.conversation()

    r1 = await convo.send("nearest cooling center to Union Square?")
    r2 = await convo.send("what about in the Bronx?")

    assert r1.text == "answer 1"
    assert r2.text == "answer 2"
    # Second turn's prompt includes the first user msg + first assistant reply + new user msg
    second_turn = seen_messages[1]
    contents = [m["content"] for m in second_turn]
    assert "nearest cooling center to Union Square?" in contents
    assert "answer 1" in contents
    assert "what about in the Bronx?" in contents
    # History accumulates: 2 user + 2 assistant
    assert len(convo.turns) == 4


async def test_run_with_explicit_history(empty_registry):
    captured = {}

    async def cap(messages, tool_schemas):
        captured["messages"] = list(messages)  # snapshot before the loop appends
        return _assistant(content="ok")

    agent = Agent(empty_registry, tools={}, complete_fn=cap)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    await agent.run("follow up", history=history)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_hits_max_iters(empty_registry):
    # Always returns a tool call → never terminates on its own.
    async def noop(args, ctx):
        return "ok"

    tool = Tool(name="loop", description="x", parameters={"type": "object", "properties": {}}, handler=noop)

    async def always_tool(messages, tool_schemas):
        return _assistant(tool_calls=[_tool_call("loop", {})])

    agent = Agent(empty_registry, tools={"loop": tool}, complete_fn=always_tool)
    result = await agent.run("go", max_iters=3)
    assert result.hit_max_iters
    assert result.iterations == 3


async def test_agent_captures_token_usage_from_stream():
    from heynyc.core.agent import Agent
    from heynyc.core.registry import Registry

    async def fake_stream(messages, tool_schemas):
        yield {"type": "text", "text": "Hello"}
        yield {"type": "usage", "input_tokens": 42, "output_tokens": 7}
        yield {"type": "message", "message": {"role": "assistant", "content": "Hello", "tool_calls": None}}

    agent = Agent(Registry([]), stream_fn=fake_stream)
    result = await agent.run("hi")
    assert result.usage["input_tokens"] == 42
    assert result.usage["output_tokens"] == 7
    assert result.usage["latency_ms"] >= 0.0
