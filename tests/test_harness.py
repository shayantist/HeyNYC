from __future__ import annotations

import pytest

from heynyc.core import events
from heynyc.core.agent import Agent, _with_retry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


@pytest.fixture
def empty_registry():
    return Registry([])


def _text(s):
    return {"type": "text", "text": s}


def _message(content, tool_calls=None):
    return {"type": "message", "message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}


def _tool_call(name, args=None, call_id="c1"):
    import json

    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args or {})}}


def _scripted_stream(*responses):
    state = {"i": 0}

    async def sf(messages, schemas):
        chunks = responses[state["i"]]
        state["i"] += 1
        for c in chunks:
            yield c

    return sf


async def test_stream_emits_text_then_done(empty_registry):
    sf = _scripted_stream([_text("Hello "), _text("there"), _message("Hello there")])
    agent = Agent(empty_registry, tools={}, stream_fn=sf)
    evs = [e async for e in agent.stream("hi")]
    types = [e.type for e in evs]
    assert types == ["message.start", "text.delta", "text.delta", "message.completed", "done"]
    done = evs[-1]
    assert isinstance(done, events.Done)
    assert done.status == "success"
    assert done.result.text == "Hello there"


async def test_stream_tool_lifecycle(empty_registry):
    async def echo(args, ctx: ToolContext):
        return "tool ran"

    tool = Tool(name="echo", description="x", parameters={"type": "object", "properties": {}}, handler=echo)
    sf = _scripted_stream(
        [_message(None, [_tool_call("echo")])],
        [_text("all done"), _message("all done")],
    )
    agent = Agent(empty_registry, tools={"echo": tool}, stream_fn=sf)
    evs = [e async for e in agent.stream("go")]
    types = [e.type for e in evs]
    assert "tool.start" in types
    completed = [e for e in evs if e.type == "tool.completed"][0]
    assert completed.status == "ok"
    assert evs[-1].result.tool_calls_made == ["echo"]


async def test_reminders_emitted_and_injected(empty_registry):
    captured = {}

    async def sf(messages, schemas):
        captured["messages"] = list(messages)
        yield _message("ok")

    agent = Agent(empty_registry, tools={}, stream_fn=sf)
    evs = [e async for e in agent.stream("q", reminders=["Today is 2026-06-25", "Location: Astoria"])]
    reminders = [e for e in evs if e.type == "reminder"]
    assert [r.summary for r in reminders] == ["Today is 2026-06-25", "Location: Astoria"]
    reminder_msgs = [m for m in captured["messages"] if "<system-reminder>" in (m.get("content") or "")]
    assert len(reminder_msgs) == 2


async def test_model_error_yields_error_and_done(empty_registry):
    async def boom(messages, schemas):
        raise RuntimeError("api down")
        yield  # noqa — makes this an async generator

    agent = Agent(empty_registry, tools={}, stream_fn=boom)
    evs = [e async for e in agent.stream("q")]
    assert any(e.type == "error" for e in evs)
    done = evs[-1]
    assert done.type == "done" and done.status == "error"


async def test_approval_denied_without_approver(empty_registry):
    async def write(args, ctx):
        return "wrote it"

    tool = Tool(
        name="write", description="x", parameters={"type": "object", "properties": {}},
        handler=write, read_only=False, requires_approval=True,
    )
    sf = _scripted_stream([_message(None, [_tool_call("write")])], [_message("ok")])
    agent = Agent(empty_registry, tools={"write": tool}, stream_fn=sf)
    evs = [e async for e in agent.stream("go")]
    assert any(e.type == "tool.approval_required" for e in evs)
    completed = [e for e in evs if e.type == "tool.completed"][0]
    assert completed.status == "error"  # denied


async def test_approval_granted_with_approver(empty_registry):
    async def write(args, ctx):
        return "wrote it"

    async def approver(name, args):
        return True

    tool = Tool(
        name="write", description="x", parameters={"type": "object", "properties": {}},
        handler=write, read_only=False, requires_approval=True,
    )
    sf = _scripted_stream([_message(None, [_tool_call("write")])], [_message("ok")])
    agent = Agent(empty_registry, tools={"write": tool}, stream_fn=sf, approver=approver)
    evs = [e async for e in agent.stream("go")]
    completed = [e for e in evs if e.type == "tool.completed"][0]
    assert completed.status == "ok"


async def test_with_retry_succeeds_after_failures():
    state = {"n": 0}

    async def factory():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert await _with_retry(factory, attempts=3, base_delay=0) == "ok"
    assert state["n"] == 3


async def test_with_retry_gives_up():
    async def factory():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError):
        await _with_retry(factory, attempts=2, base_delay=0)


async def test_community_web_result_carries_disclaimer_to_model(empty_registry):
    """Integration guarantee (§16.3): when web_search surfaces a community-tier result,
    the ⚠️ disclaimer reaches the model's context through the tool lifecycle — deterministic,
    no live Tavily. (The isolated tool is unit-tested in test_web_search.py.)"""
    from heynyc.core.tools.web_search import web_search_tools

    async def fake_search(query, allowed, recency=None):
        return [{"title": "User meetup", "url": "https://eventbrite.com/e/x", "snippet": "posted by a user"}]

    ws = web_search_tools(
        ["eventbrite.com"], source_tiers={"eventbrite.com": ("community", "events")}, search_fn=fake_search
    )[0]  # the default web_search (recent_developments is the sibling recency tool)
    sf = _scripted_stream(
        [_message(None, [_tool_call("web_search", {"query": "any meetups?"})])],
        [_text("Here's what I found"), _message("Here's what I found")],
    )
    agent = Agent(empty_registry, tools={"web_search": ws}, stream_fn=sf)
    evs = [e async for e in agent.stream("any community meetups this weekend?")]
    completed = [e for e in evs if e.type == "tool.completed"][0]
    assert completed.status == "ok"
    assert "⚠️" in completed.result_summary
    assert "confirm before you go" in completed.result_summary.lower()


def test_repl_segments_preserve_chronological_order():
    """REPL render order is a stack: preamble text, then the tools it triggered, then the
    answer — a tool note must NOT float above text that streamed before it."""
    from heynyc.__main__ import _append_segment

    segs: list = []
    _append_segment(segs, "text", "Let me look ")
    _append_segment(segs, "text", "that up!")           # consecutive text merges
    _append_segment(segs, "tool", "· using nearest…")   # a tool breaks the text block
    _append_segment(segs, "tool", "· using geocode…")
    _append_segment(segs, "text", "Here are the results")  # answer is its own block, below the tools
    assert [(s["kind"], s["text"]) for s in segs] == [
        ("text", "Let me look that up!"),
        ("tool", "· using nearest…"),
        ("tool", "· using geocode…"),
        ("text", "Here are the results"),
    ]


def test_to_sse_formats_frame():
    frame = events.to_sse(events.TextDelta(message_id="m0", text="hi"))
    assert frame.startswith("event: text.delta\n")
    assert '"text": "hi"' in frame
    assert frame.endswith("\n\n")


def test_stats_summary_renders(tmp_path, capsys):
    # the stats command summarizes a telemetry log without touching the network
    from heynyc.core import telemetry
    from heynyc.__main__ import _render_stats

    path = tmp_path / "telemetry.jsonl"
    telemetry.record_turn(path, session_id="s", model="m",
                          usage={"input_tokens": 100, "output_tokens": 40, "latency_ms": 250.0},
                          n_tool_calls=1, tool_names=["benefits_search"], status="success")
    _render_stats(path)
    out = capsys.readouterr().out
    assert "1" in out          # one turn
    assert "benefits_search" in out
