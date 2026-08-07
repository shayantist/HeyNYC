"""The console channel: the interactive REPL runs the SAME orchestrator path a texter rides
(dedup, rate-limit, encrypted persistent sessions, the free commands, spend cap, identity),
differing only in presentation. `Agent.run`'s `event_sink` drives a transient live view (streamed
text, tool notes, a thinking spinner); the settled reply is the channel-rendered markdown, printed
by rich. Presentation lives here; every guard and all accounting stay in the shared orchestrator."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Optional

from heynyc.core import config, events
from heynyc.core.agent import Agent
from heynyc.core.drafts import DraftStore
from heynyc.core.registry import Registry

from .base import KeyedLocks
from .orchestrator import Deps
from .store import ChannelStore

# A deterministic salt for local dev so the REPL keys a `console:<user>` session without requiring
# HEYNYC_PII_SALT (which serving still requires). Local only, never a production pseudonymity claim.
_LOCAL_DEV_SALT = "heynyc-local-dev"


# --- ordered-segment machinery (moved from __main__; the frozen `--raw` REPL imports it back) ---

def _append_segment(segments: list, kind: str, text: str) -> None:
    """Accumulate a stream event into ordered render segments.

    Consecutive text deltas merge into one block; a tool note breaks the text so ordering is
    preserved, a tool call that arrives after some preamble text renders *below* it (a chronological
    stack), not pinned above the whole message."""
    if kind == "text" and segments and segments[-1]["kind"] == "text":
        segments[-1]["text"] += text
    else:
        segments.append({"kind": kind, "text": text})


def _reconcile_message_text(segments: list, start: int, text: str) -> None:
    """Replace streamed deltas with the authoritative completed-message snapshot."""
    del segments[start:]
    if text.strip():
        _append_segment(segments, "text", text)


def _approve_repl_action(console, name: str, args: dict) -> bool:
    """Ask for local consent without printing the tool arguments, which may contain PII."""
    if name == "prepare_snap_application":
        action = (
            "create the local SNAP PDF from the answers you reviewed"
            if args.get("confirmed")
            else "save these answers in a local draft and show you a review"
        )
        prompt = f"[bold yellow]Allow HeyNYC to {action}? It will not submit anything. [y/N] [/]"
    else:
        prompt = f"[bold yellow]Allow HeyNYC to run {name}? [y/N] [/]"
    return console.input(prompt).strip().lower() in {"y", "yes"}


class ConsoleReplier:
    """The console `Replier`: markdown-render replies with rich, and persist any tool-produced
    document into the repl-artifacts dir. `send_document` COPIES the file BEFORE returning because
    the orchestrator rmtree's its per-request tempdir the moment `handle` completes."""

    def __init__(self, console, artifacts_dir: Path) -> None:
        self._console = console
        self._artifacts_dir = Path(artifacts_dir)

    async def send_text(self, text: str) -> None:
        from rich.markdown import Markdown

        self._console.print(Markdown(text))

    async def indicate_typing(self) -> None:
        return None   # the streaming view (ConsoleSink) is the console's "typing" signal

    async def send_document(self, path: str, caption: str = "") -> None:
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        dest = self._artifacts_dir / Path(path).name
        shutil.copy2(path, dest)   # copy NOW: the orchestrator tempdir is about to be removed
        self._console.print(f"[bold green]📄 saved your filled draft →[/] {dest}")


class ConsoleSink:
    """Renders a turn's event stream as a transient live view: streamed text, tool notes, and a
    thinking spinner, in chronological order. Passed to `Agent.run` as its `event_sink`. The view
    is transient (cleared when the turn's `Done` arrives); the settled reply is the channel-rendered
    markdown `ConsoleReplier` prints afterward. A pure observer: its exceptions are swallowed by run,
    so a rendering bug never aborts a turn."""

    def __init__(self, console) -> None:
        self._console = console
        self._live = None
        self._segments: list = []
        self._message_start = 0
        self._done = False
        self.tools_used: list[str] = []

    def start_turn(self) -> None:
        self._segments = []
        self._message_start = 0
        self._done = False
        self._live = None
        self.tools_used = []

    def finish(self) -> None:
        """Safety net: stop the live view if a turn ended without a `Done` (it never should)."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __call__(self, event) -> None:
        if self._live is None and not self._done:
            self._start_live()
        if isinstance(event, events.MessageStart):
            self._message_start = len(self._segments)
        elif isinstance(event, events.ToolStart):
            self.tools_used.append(event.name)
            _append_segment(self._segments, "tool", f"· using {event.name}…")
        elif isinstance(event, events.TextDelta):
            _append_segment(self._segments, "text", event.text)
        elif isinstance(event, events.MessageCompleted):
            _reconcile_message_text(self._segments, self._message_start, event.text)
        elif isinstance(event, events.Done):
            self._done = True
        # Reminder / ToolApprovalRequired render nothing here: reminders are internal, and approval
        # is handled by the agent's approver, which prompts on this same console directly.
        if self._live is not None:
            self._live.update(self._render())
            if self._done:
                self._live.stop()   # transient: clears the preview, the settled reply prints next
                self._live = None
                if self.tools_used:
                    # The transient view vanishes with the Live; keep a persistent one-line
                    # record of the tools this turn used, above the settled reply (owner ask).
                    seen = list(dict.fromkeys(self.tools_used))
                    self._console.print(f"[dim]· used: {', '.join(seen)}[/]")
                    self.tools_used = []
                self._live = None

    def _start_live(self) -> None:
        from rich.live import Live

        self._live = Live(
            self._render(), console=self._console, refresh_per_second=12,
            transient=True, vertical_overflow="visible",
        )
        self._live.start()

    def _render(self):
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.spinner import Spinner
        from rich.text import Text

        parts: list = []
        for seg in self._segments:
            if seg["kind"] == "tool":
                parts.append(Text(seg["text"], style="dim cyan"))
            elif seg["text"].strip():
                parts.append(Markdown(seg["text"]))
        # Spinner while we wait (before any output, or after a tool note until the answer starts
        # streaming), hidden once answer text is flowing or the turn is done.
        answering = (
            bool(self._segments)
            and self._segments[-1]["kind"] == "text"
            and self._segments[-1]["text"].strip()
        )
        if not self._done and not answering:
            parts.append(Spinner("dots", text=Text(" thinking…", style="dim")))
        return Group(*parts)


def _load_retriever(data_dir: Path):
    from heynyc.core.index import IndexRetriever, default_embedder, open_store

    index_path = Path(data_dir) / "index.lance"
    if not index_path.exists():
        return None
    return IndexRetriever(store=open_store(index_path), embedder=default_embedder())


def build_console_deps(*, console, model: Optional[Any] = None, data_dir: Optional[Path] = None) -> Deps:
    """Assemble the console channel's `Deps`, mirroring `app.build_deps` but with a console-native
    agent. The legacy runtime carries the local approver; the Pydantic runtime uses the shared
    durable channel approval flow. Both honor the `--model` override. `--user` identity + the
    local-dev salt are applied by the caller at message time. `event_sink` is the live-view
    `ConsoleSink`; every other channel leaves it None."""
    from heynyc.modules.advisories.tools import current_awareness

    data = Path(data_dir) if data_dir is not None else config.HEYNYC_DATA_DIR

    def approver_sync(name: str, args: dict) -> bool:
        return _approve_repl_action(console, name, args)

    async def approver(name: str, args: dict) -> bool:
        return approver_sync(name, args)

    registry = Registry.discover(
        config.MODULES_DIR,
        config.BASE_ALLOWLIST,
        config.NEWS_ALLOWLIST,
    )
    selected_model = model or config.HEYNYC_MODEL
    index = _load_retriever(data)
    if config.HEYNYC_AGENT_RUNTIME == "pydantic":
        from heynyc.core.pydantic_runtime import build_configured_runtime

        agent = build_configured_runtime(
            registry,
            model=selected_model,
            index=index,
            current_awareness=current_awareness,
        )
    else:
        agent = Agent(
            registry,
            model=selected_model,
            index=index,
            approver=approver,
            notify_awareness=current_awareness,
            scope_gate=True,
        )
    store = ChannelStore(
        data / "channels.sqlite3", rate_limit=config.CHANNEL_RATE_LIMIT,
        window_s=config.CHANNEL_RATE_WINDOW_S, dedup_ttl_s=config.CHANNEL_DEDUP_TTL_S,
    )
    if not hasattr(agent, "conversation_from_state"):
        store.clear_pending_approvals()
    from heynyc.core import telemetry

    return Deps(
        agent=agent, store=store, sessions_dir=data / "sessions",
        salt=config.HEYNYC_PII_SALT or _LOCAL_DEV_SALT,
        telemetry_path=telemetry.default_path(data), feedback_path=data / "feedback.jsonl",
        locks=KeyedLocks(), semaphore=asyncio.Semaphore(config.CHANNEL_MAX_CONCURRENCY),
        drafts=DraftStore(data / "drafts"),
        user_daily_spend_cap=config.HEYNYC_USER_DAILY_SPEND_CAP,
        event_sink=ConsoleSink(console),
    )
