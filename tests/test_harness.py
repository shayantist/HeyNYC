from __future__ import annotations

import pytest

from heynyc.core import events
from heynyc.core.agent import Agent, AgentResult, _with_retry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.runner import merge_eval_results


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


def test_merge_eval_results_keeps_pending_and_final_diagnostics():
    pending = AgentResult(
        text="",
        citations={},
        diagnostics={"validation_rejections": [{"stage": "pending"}]},
    )
    final = AgentResult(
        text="done",
        citations={},
        diagnostics={"validation_rejections": [{"stage": "final"}]},
    )

    result = merge_eval_results(pending, final, set())

    assert result.diagnostics == {
        "validation_rejections": [
            {"stage": "pending"},
            {"stage": "final"},
        ]
    }


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


async def test_unknown_citation_marker_is_rejected_and_regenerated(empty_registry):
    raw = "Draft ready {cite:S5} {cite:prepare_snap_application}"
    sf = _scripted_stream(
        [_text(raw), _message(raw)],
        [_text("Draft ready"), _message("Draft ready")],
    )
    agent = Agent(empty_registry, tools={}, stream_fn=sf)
    evs = [e async for e in agent.stream("go")]

    completed = [e for e in evs if e.type == "message.completed"]
    assert [event.text for event in completed] == ["", "Draft ready"]
    assert sum(e.type == "message.start" for e in evs) == 2
    assert evs[-1].result.text == "Draft ready"


async def test_discovery_citation_marker_is_rejected_and_regenerated(empty_registry):
    async def discovery(_args, ctx: ToolContext):
        cite_id = ctx.citations.register(
            "https://www.nyc.gov/search-result",
            snippet="A truncated search result",
            kind="WEB",
            provenance={"evidence_grade": "discovery"},
        )
        return f"A truncated search result {{cite:{cite_id}}}"

    raw = "The snippet proves more than it says. {cite:S1}"
    sf = _scripted_stream(
        [_message(None, [_tool_call("discovery")])],
        [_text(raw), _message(raw)],
        [_text("I couldn't verify that detail."), _message("I couldn't verify that detail.")],
    )
    agent = Agent(
        empty_registry,
        tools={
            "discovery": Tool(
                name="discovery",
                description="Search for a source",
                parameters={"type": "object", "properties": {}},
                handler=discovery,
            ),
        },
        stream_fn=sf,
    )

    evs = [event async for event in agent.stream("Can you confirm it?")]

    assert evs[-1].result.text == "I couldn't verify that detail."


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
    # 3 injected reminders: the volatile now-line/blurbs (cache-layout fix) plus the two explicit
    # ones; only the two explicit reminders surface as Reminder events (asserted above)
    assert len(reminder_msgs) == 3


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

    async def fake_search(query, allowed, published_after=None, published_before=None, count=5):
        return [{"title": "User meetup", "url": "https://eventbrite.com/e/x", "snippet": "posted by a user"}]

    ws = web_search_tools(
        ["eventbrite.com"], source_tiers={"eventbrite.com": ("community", "events")}, search_fn=fake_search
    )[0]  # the default web_search, including its optional publication bounds
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
    from heynyc.channels.console import _append_segment

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


def test_repl_reconciles_streamed_text_with_completed_message():
    from heynyc.channels.console import _append_segment, _reconcile_message_text

    segs = [{"kind": "tool", "text": "· using prepare_snap_application…"}]
    start = len(segs)
    _append_segment(segs, "text", "Draft ready {cite:S5}")
    _reconcile_message_text(segs, start, "Draft ready")

    assert segs == [
        {"kind": "tool", "text": "· using prepare_snap_application…"},
        {"kind": "text", "text": "Draft ready"},
    ]


def test_repl_snap_approval_is_explicit_and_does_not_echo_pii():
    from heynyc.channels.console import _approve_repl_action

    class Console:
        def __init__(self, answer):
            self.answer = answer
            self.prompts = []

        def input(self, prompt):
            self.prompts.append(prompt)
            return self.answer

    args = {"slots": {"legal_name": "Ana Diaz", "ssn": "078-05-1120"}, "confirmed": True}
    denied = Console("")
    approved = Console("yes")

    assert _approve_repl_action(denied, "prepare_snap_application", args) is False
    assert _approve_repl_action(approved, "prepare_snap_application", args) is True
    prompt = " ".join(denied.prompts + approved.prompts)
    assert "Ana Diaz" not in prompt and "078-05-1120" not in prompt
    assert "will not submit" in prompt.lower()


def test_repl_keyboard_interrupt_exits_without_traceback(monkeypatch):
    from heynyc.__main__ import _run_repl

    def interrupted(_coroutine):
        _coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("heynyc.__main__.asyncio.run", interrupted)
    _run_repl()


def test_repl_screening_requires_explicit_command():
    from heynyc.__main__ import _screen_turn_options

    assert _screen_turn_options("I need food help") == {
        "forced_tool": None,
        "forced_tool_args": None,
        "excluded_tools": {"screen_access_nyc_eligibility"},
        "screen_reminder": None,
    }
    assert _screen_turn_options("/screen")["forced_tool_args"] == {"show_all": False}
    assert _screen_turn_options(" /SCREEN ALL ")["forced_tool_args"] == {"show_all": True}


def test_eval_case_filter_selects_requested_ids_and_rejects_typos():
    from heynyc.eval.cases import EvalCase, select_cases

    cases = [
        EvalCase(id="benefits_snap", module="benefits", query="q"),
        EvalCase(id="events_weekend", module="events", query="q"),
    ]

    assert [case.id for case in select_cases(cases, case_ids=["events_weekend"])] == [
        "events_weekend"
    ]
    with pytest.raises(SystemExit, match="Unknown eval case id: typo"):
        select_cases(cases, case_ids=["typo"])


def test_eval_run_metadata_preserves_usage_cost_and_case_ids():
    from types import SimpleNamespace

    from heynyc.__main__ import _eval_run_metadata

    results = [
        SimpleNamespace(case=SimpleNamespace(id="a"), usage={
            "input_tokens": 10, "output_tokens": 2, "cost_usd": 0.01,
            "latency_ms": 100, "n_model_calls": 2, "n_tool_calls": 1,
        }),
        SimpleNamespace(case=SimpleNamespace(id="b"), usage={
            "input_tokens": 20, "output_tokens": 3, "cost_usd": 0.02,
            "latency_ms": 200, "n_model_calls": 3, "n_tool_calls": 2,
        }),
    ]

    assert _eval_run_metadata("model", results) == {
        "model": "model", "case_ids": ["a", "b"],
        "input_tokens": 30, "output_tokens": 5, "candidate_cost_usd": 0.03,
        "latency_ms": 300.0, "n_model_calls": 5, "n_tool_calls": 3,
    }


def test_console_turn_records_pii_free_telemetry_tagged_console(tmp_path):
    """The unified console REPL records through the SAME shared analytics path a texter does,
    tagged channel='console' and keyed off the salted user_key (never a raw identity)."""
    from types import SimpleNamespace

    from heynyc.channels import analytics
    from heynyc.core import telemetry

    result = SimpleNamespace(
        usage={"latency_ms": 250.0, "tool_time_ms": 40.0},
        tool_calls_made=["screen_access_nyc_eligibility"],
        citations={},
        diagnostics={},
        status="success",
        text="here you go",
    )

    analytics.record_interaction(
        telemetry_path=tmp_path / "t.jsonl", model="openai/gpt-5.4-mini",
        user_key="consolekey", channel="console", result=result,
    )

    records = telemetry.load(tmp_path / "t.jsonl")
    assert records[-1]["channel"] == "console"
    assert records[-1]["session_id"] == "consolekey"
    assert records[-1]["latency_ms"] == 250.0
    assert records[-1]["tool_names"] == ["screen_access_nyc_eligibility"]


def test_to_sse_formats_frame():
    frame = events.to_sse(events.TextDelta(message_id="m0", text="hi"))
    assert frame.startswith("event: text.delta\n")
    assert '"text": "hi"' in frame
    assert frame.endswith("\n\n")


def test_stats_summary_renders(tmp_path, capsys):
    # the stats command summarizes a telemetry log without touching the network
    from heynyc.__main__ import _render_stats
    from heynyc.core import telemetry

    path = tmp_path / "telemetry.jsonl"
    telemetry.record_turn(path, session_id="s", model="m",
                          usage={"input_tokens": 100, "output_tokens": 40, "latency_ms": 250.0},
                          n_tool_calls=1, tool_names=["benefits_search"], status="success")
    _render_stats(path)
    out = capsys.readouterr().out
    assert "1" in out          # one turn
    assert "benefits_search" in out


def test_case_listing_renders_one_greppable_row_per_case():
    """The audit surface: every case in the corpus on one line: id, source file, flags, tags."""
    from heynyc.core import config
    from heynyc.core.registry import Registry
    from heynyc.eval.cases import render_case_listing

    registry = Registry.discover(config.MODULES_DIR)
    out = render_case_listing(registry)
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) > 200                      # the whole corpus, one row each
    assert any("housing_no_heat" in l and "F063" in l for l in lines)
    assert any("convo_past_tense_identity_not_trivia" in l and "global" in l for l in lines)
    header = lines[0]
    assert "id" in header and "tags" in header


def test_repl_temp_flag_isolates_and_randomizes(monkeypatch):
    """--temp: a throwaway data dir and a random identity, so test sessions persist nothing.
    Pins the seams: the parser accepts the flag, and build_console_deps honors data_dir by
    placing the store inside it (the whole channel stack then writes into the grave)."""
    import tempfile
    from pathlib import Path

    from pydantic_ai.models.test import TestModel
    from rich.console import Console

    from heynyc.channels.console import build_console_deps

    with tempfile.TemporaryDirectory() as td:
        deps = build_console_deps(console=Console(), model=TestModel(), data_dir=Path(td))
        assert str(Path(td)) in str(deps.sessions_dir)
    import heynyc.__main__ as cli
    src = __import__("inspect").getsource(cli._cmd_repl)
    assert "TemporaryDirectory" in src and 'f"temp-{uuid.uuid4().hex[:8]}"' in src
    assert "temp_dir.cleanup()" in src


def test_unwrap_docs_joins_prose_and_preserves_structure():
    """The unwrap tool: prose joins to full lines; fences, tables, lists, headers, hard breaks,
    and blank lines are byte-preserved."""
    import sys
    sys.path.insert(0, "scripts")
    from unwrap_docs import unwrap_text

    src = (
        "# Title\n\nThis line was wrapped\nat a fixed width\nlong ago.\n\n"
        "- a list item\n- another\n\n| a | b |\n|---|---|\n\n```\ncode stays\nwrapped\n```\n\n"
        "A hard break  \nstays broken.\n"
    )
    out = unwrap_text(src)
    assert "This line was wrapped at a fixed width long ago." in out
    assert "- a list item\n- another" in out
    assert "| a | b |\n|---|---|" in out
    assert "```\ncode stays\nwrapped\n```" in out
    assert "A hard break  \nstays broken." in out
    assert out.count("# Title") == 1

    # Badge lines (the README idiom of stacked image links) are deliberate structure, not
    # fixed-width wrapping; they must never be joined into one line.
    badges = "[![CI](https://x/ci.svg)](https://x/ci)\n[![License](https://x/l.svg)](LICENSE)\n\nprose here\nwrapped once.\n"
    out2 = unwrap_text(badges)
    assert "[![CI](https://x/ci.svg)](https://x/ci)\n[![License](https://x/l.svg)](LICENSE)" in out2
    assert "prose here wrapped once." in out2
