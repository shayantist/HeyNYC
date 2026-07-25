"""The console channel: the REPL now rides the SAME orchestrator path a texter does, differing
only in presentation. These pin the presentation seam (ConsoleReplier, ConsoleSink) and that the
free commands / welcome / consent flows reach the console key through build_console_deps + handle."""
from __future__ import annotations

import shutil
from io import StringIO

from heynyc.channels.base import InboundMessage
from heynyc.channels.identity import user_key
from heynyc.channels.orchestrator import handle
from heynyc.core import events
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry


def _recording_console():
    from rich.console import Console

    return Console(file=StringIO(), record=True, width=200, force_terminal=False)


def _scripted_agent(reply="Here you go."):
    async def complete_fn(messages, tool_schemas):
        return {"role": "assistant", "content": reply, "tool_calls": None}

    return Agent(Registry([]), tools={}, complete_fn=complete_fn, model="fake-model")


def _console_msg(text, mid, user="local"):
    return InboundMessage(channel="console", sender=user, text=text, message_id=mid)


# --- ConsoleReplier -------------------------------------------------------------------------

async def test_console_send_document_copies_before_tempdir_teardown(tmp_path):
    """The orchestrator rmtree's its per-request tempdir in `finally`; the console replier must
    COPY the PDF into the persistent repl-artifacts dir BEFORE returning, or the draft is lost."""
    from heynyc.channels.console import ConsoleReplier

    artifacts = tmp_path / "repl-artifacts"
    artifacts.mkdir()
    tempdir = tmp_path / "heynyc-art-xyz"
    tempdir.mkdir()
    pdf = tempdir / "LDSS-4826.pdf"
    pdf.write_bytes(b"%PDF-1.4 filled draft")

    replier = ConsoleReplier(_recording_console(), artifacts)
    await replier.send_document(str(pdf), caption="Your draft SNAP application")

    shutil.rmtree(tempdir)   # exactly what the orchestrator does after handle returns

    copied = artifacts / "LDSS-4826.pdf"
    assert copied.exists()
    assert copied.read_bytes() == b"%PDF-1.4 filled draft"


async def test_console_send_text_renders_markdown(tmp_path):
    from heynyc.channels.console import ConsoleReplier

    console = _recording_console()
    replier = ConsoleReplier(console, tmp_path)
    await replier.send_text("**Bold** and a point")
    out = console.export_text()
    assert "Bold and a point" in out          # rich rendered the markdown (delimiters consumed)


# --- ConsoleSink ----------------------------------------------------------------------------

def test_console_sink_orders_tools_below_preamble_and_reconciles_final():
    from heynyc.channels.console import ConsoleSink

    # The real stream for a preamble + tool + answer turn: the tool-call message reconciles its
    # preamble (MessageCompleted) BEFORE the tools, then a fresh MessageStart precedes the answer.
    sink = ConsoleSink(_recording_console())
    sink.start_turn()
    sink(events.MessageStart(message_id="m0"))
    sink(events.TextDelta(message_id="m0", text="Let me look "))
    sink(events.TextDelta(message_id="m0", text="that up."))
    sink(events.MessageCompleted(message_id="m0", text="Let me look that up."))
    sink(events.ToolStart(tool_call_id="c1", name="nearest"))
    sink(events.MessageStart(message_id="m1"))
    sink(events.TextDelta(message_id="m1", text="Here are the results {cite:S1}"))
    sink(events.MessageCompleted(message_id="m1", text="Here are the results"))
    segments = [(s["kind"], s["text"]) for s in sink._segments]
    sink.finish()

    assert segments == [
        ("text", "Let me look that up."),
        ("tool", "· using nearest…"),
        ("text", "Here are the results"),
    ]


def test_console_sink_ignores_reminder_and_approval_events():
    from heynyc.channels.console import ConsoleSink

    sink = ConsoleSink(_recording_console())
    sink.start_turn()
    sink(events.MessageStart(message_id="m0"))
    sink(events.Reminder(summary="today is..."))
    sink(events.ToolApprovalRequired(tool_call_id="c1", name="prepare_snap_application", args={}))
    sink(events.TextDelta(message_id="m0", text="ok"))
    segments = [(s["kind"], s["text"]) for s in sink._segments]
    sink.finish()

    assert segments == [("text", "ok")]   # no segment for the reminder or the approval prompt


# --- build_console_deps + the free commands through the real path ----------------------------

def _console_deps(tmp_path, *, model=None):
    from heynyc.channels.console import build_console_deps

    return build_console_deps(console=_recording_console(), model=model, data_dir=tmp_path)


def test_build_console_deps_populates_every_dep_and_wires_the_approver(tmp_path, monkeypatch):
    from heynyc.channels.console import ConsoleSink
    from heynyc.core import config

    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "legacy")
    deps = _console_deps(tmp_path)
    assert deps.agent is not None
    assert deps.store is not None
    assert deps.sessions_dir == tmp_path / "sessions"
    assert deps.salt                                    # a local-dev fallback when none is set
    assert deps.telemetry_path and deps.feedback_path
    assert deps.locks is not None and deps.semaphore is not None
    assert deps.drafts is not None
    assert isinstance(deps.event_sink, ConsoleSink)
    assert deps.agent._approver is not None             # forms would auto-deny without it


def test_console_model_flag_reaches_the_console_agent(tmp_path, monkeypatch):
    from heynyc.core import config

    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "legacy")
    deps = _console_deps(tmp_path, model="openai/some-other-model")
    assert deps.agent.model == "openai/some-other-model"


def test_console_uses_the_configured_pydantic_runtime(tmp_path, monkeypatch):
    from heynyc.core import config

    sentinel = object()
    seen = {}

    def build(registry, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "pydantic")
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_configured_runtime",
        build,
    )

    deps = _console_deps(tmp_path, model="openai/test-model")

    assert deps.agent is sentinel
    assert seen["model"] == "openai/test-model"
    assert seen["current_awareness"] is not None


def test_legacy_console_startup_invalidates_pydantic_approvals(tmp_path, monkeypatch):
    from heynyc.channels.store import ChannelStore
    from heynyc.core import config, pii_crypto

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = ChannelStore(
        tmp_path / "channels.sqlite3",
        rate_limit=20,
        window_s=60,
        dedup_ttl_s=3600,
    )
    store.set_pending_approval("resident", b'{"pending":true}', ttl_s=60)
    monkeypatch.setattr(config, "HEYNYC_AGENT_RUNTIME", "legacy")

    deps = _console_deps(tmp_path)

    assert deps.store.has_pending_approval("resident") is False


async def test_console_greeting_gets_the_help_menu(tmp_path):
    deps = _console_deps(tmp_path)
    console = deps.event_sink._console
    deps.agent = _scripted_agent()   # if the agent ran, we'd see the answer, not the menu

    await handle(_console_msg("hey", "g1"), _ConsoleReplierFor(deps, console), deps)

    out = console.export_text()
    assert "REPORT" in out and "ask me" in out          # the grounded capability menu
    assert "Here you go." not in out                     # the greeting short-circuited the agent


async def test_console_welcome_footer_appears_once_on_the_console_key(tmp_path):
    """Per-channel welcome (owner, 2026-07-21): the console NEVER gets the texting footer,
    the welcome LEADS the first answer on every channel (the banner lists, the welcome
    explains), then never repeats."""
    deps = _console_deps(tmp_path)
    console = deps.event_sink._console
    deps.agent = _scripted_agent()
    replier = _ConsoleReplierFor(deps, console)

    await handle(_console_msg("when do cooling centers open?", "w1"), replier, deps)
    first = console.export_text()
    assert "First time here" in first and "Now, about your message:" in first
    assert first.index("First time here") < first.index("Here you go.")  # greet, THEN answer

    console2 = _recording_console()
    replier2 = ConsoleReplierWith(console2, tmp_path)
    deps.event_sink._console = console2
    await handle(_console_msg("what about SNAP?", "w2"), replier2, deps)
    assert "First time here" not in console2.export_text()  # once ever


async def test_console_report_confirm_flow(tmp_path):
    deps = _console_deps(tmp_path)
    console = deps.event_sink._console
    deps.agent = _scripted_agent("Cooling centers open Saturday.")
    replier = _ConsoleReplierFor(deps, console)

    await handle(_console_msg("when do cooling centers open?", "r0"), replier, deps)
    await handle(_console_msg("report", "r1"), replier, deps)
    assert deps.store.flags() == []                      # nothing recorded before YES

    await handle(_console_msg("YES", "r2"), replier, deps)
    assert len(deps.store.flags()) == 1                  # exactly one pointer, on confirm


async def test_console_delete_confirm_flow(tmp_path):
    deps = _console_deps(tmp_path)
    console = deps.event_sink._console
    deps.agent = _scripted_agent("Cooling centers open Saturday.")
    replier = _ConsoleReplierFor(deps, console)
    key = user_key("console", "local", deps.salt)

    await handle(_console_msg("when do cooling centers open?", "d0"), replier, deps)
    session_file = deps.sessions_dir / f"{key}.jsonl"
    assert session_file.exists()

    await handle(_console_msg("DELETE MY DATA", "d1"), replier, deps)
    assert session_file.exists()                         # staged only, nothing deleted yet
    await handle(_console_msg("YES", "d2"), replier, deps)
    assert not session_file.exists()                     # the transcript is actually gone


# --- test helpers: a ConsoleReplier bound to a chosen console --------------------------------

def ConsoleReplierWith(console, artifacts_dir):
    from heynyc.channels.console import ConsoleReplier

    return ConsoleReplier(console, artifacts_dir)


def _ConsoleReplierFor(deps, console):
    from heynyc.channels.console import ConsoleReplier

    return ConsoleReplier(console, deps.sessions_dir.parent / "repl-artifacts")


def test_sink_prints_persistent_tool_line_above_settled_reply():
    """The transient live view vanishes on Done; the tools a turn used stay visible as one dim
    line printed before the settled reply (owner ask: see what tools ran, after the fact)."""
    from heynyc.channels.console import ConsoleSink
    from heynyc.core import events as ev

    console = _recording_console()
    sink = ConsoleSink(console)
    sink.start_turn()
    sink(ev.ToolStart(tool_call_id="1", name="nyc_advisories", label="nyc_advisories"))
    sink(ev.ToolStart(tool_call_id="2", name="street_closures", label="street_closures"))
    sink(ev.ToolStart(tool_call_id="3", name="street_closures", label="street_closures"))
    sink(ev.Done(status="success", num_turns=1, citations={}, result=None))
    out = console.export_text()   # export CLEARS the record buffer
    assert "· used: nyc_advisories, street_closures" in out   # deduped, order kept
    sink.start_turn()
    sink(ev.Done(status="success", num_turns=1, citations={}, result=None))
    assert "· used:" not in console.export_text()             # no line on a tool-free turn
