"""HeyNYC CLI.

    uv run python -m heynyc modules           # list discovered service modules
    uv run python -m heynyc index-build        # fetch + embed module seeds into the index
    uv run python -m heynyc index-search "q"   # query the index (no LLM)
    uv run python -m heynyc chat "question"    # ask the agent (needs an LLM key)
"""
from __future__ import annotations

import argparse
import asyncio
import warnings

# LiteLLM serializes usage through a Pydantic model that predates the newer OpenAI token-detail
# fields, emitting a harmless "Pydantic serializer warnings" UserWarning on every priced call.
# Suppress exactly that message at the CLI surface so it never interleaves with resident-facing
# output (observed live mid-REPL); the usage numbers themselves flow correctly either way.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning, module="pydantic.main",
)
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load the app's .env HERE (config.py + the engine no longer auto-load it, so the core stays
# reusable). Must run before importing config, which reads env at import time.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from heynyc.core import config, events  # noqa: E402
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry

_INDEX_PATH = config.HEYNYC_DATA_DIR / "index.lance"


def _default_reminders() -> list[str]:
    """Reactive context injected each turn (date, city). Location can be added per-user."""
    return [f"Today's date is {datetime.now():%A, %B %-d, %Y}. The user is in New York City."]


def _load_retriever(required: bool):
    """Open the on-disk index as an IndexRetriever, or None if not built."""
    from heynyc.core.index import IndexRetriever, default_embedder, open_store

    if not _INDEX_PATH.exists():
        if required:
            print("No index found. Run: uv run python -m heynyc index-build")
        return None
    embedder = default_embedder()
    store = open_store(_INDEX_PATH, model_id=embedder.model_id)
    return IndexRetriever(store=store, embedder=embedder)


_MANIFEST_TEMPLATE = """\
# {name} service module. Fill in the fields below, no code needed for most services.
# Docs: heynyc/modules/README.md
name: {name}
category: general            # health | transit | housing | benefits | events | tourism | ...
description: >-
  One sentence describing what this service helps people find or do.
keywords:                     # words/phrases that should trigger this module
  - {name}
# Optional: a NYC Open Data (Socrata) dataset for "nearest X" lookups.
# Find datasets at https://data.cityofnewyork.us, copy the dataset id from its URL.
datasets: []
#  - id: xxxx-xxxx
#    category: {name}          # the category name the agent passes to nearest()
#    field_map:                # map the dataset's real column names to these keys
#      name: <name_column>
#      lat: <latitude_column>
#      lon: <longitude_column>
#      status: <status_column>   # optional
#      borough: <borough_column> # optional
#    where: ""                 # optional SoQL filter, e.g. "status='Activated'"
seeds: []                     # official pages to index for how-to / eligibility info
#  - https://www.nyc.gov/...
allowlist: []                 # extra trusted domains this module may web-search
#  - example.nyc.gov
prompt: |
  Tell the agent how to help with this service: which tool to use, what to cite,
  and when to abstain. Keep it short and concrete.
eval: eval.yaml
"""

_EVAL_TEMPLATE = """\
# Golden eval cases for the {name} module. Each is a real user question with the
# expected behavior. These run in the eval gate (Phase 5) to prove no hallucination.
- id: {name}_basic
  query: "A typical question a user would ask about {name}."
  expect_tools: []            # e.g. [nearest] or [index_search] or [web_search]
  abstain: false
  notes: What a good grounded answer looks like.

- id: {name}_out_of_scope
  query: "Something this module should NOT answer."
  abstain: true
  notes: Should decline rather than guess.
"""


def _cmd_new_module(name: str) -> None:
    from heynyc.core import config

    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    mod_dir = config.MODULES_DIR / name
    if mod_dir.exists():
        print(f"Module '{name}' already exists at {mod_dir}")
        return
    mod_dir.mkdir(parents=True)
    (mod_dir / "manifest.yaml").write_text(_MANIFEST_TEMPLATE.format(name=name))
    (mod_dir / "eval.yaml").write_text(_EVAL_TEMPLATE.format(name=name))
    print(f"Created module '{name}':")
    print(f"  {mod_dir / 'manifest.yaml'}  ← edit this")
    print(f"  {mod_dir / 'eval.yaml'}      ← add a few test questions")
    print("\nThen run:  uv run python -m heynyc modules   to confirm it loads.")
    print("See heynyc/modules/README.md for the full guide.")


CAPABILITIES_START = "<!-- CAPABILITIES:START -->"
CAPABILITIES_END = "<!-- CAPABILITIES:END -->"


def _readme_path() -> Path:
    return config.PROJECT_ROOT / "README.md"


def _write_readme_capabilities(markdown: str) -> bool:
    """Rewrite just the content between the CAPABILITIES markers in README.md.

    Idempotent: returns True only when the file actually changed. A function
    replacement is used so backslash escapes in the generated table (e.g. an
    escaped pipe) are never treated as regex backreferences."""
    import re

    path = _readme_path()
    text = path.read_text()
    block = f"{CAPABILITIES_START}\n\n{markdown}\n\n{CAPABILITIES_END}"
    pattern = re.compile(
        re.escape(CAPABILITIES_START) + r".*?" + re.escape(CAPABILITIES_END), re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(
            f"README markers not found. Add {CAPABILITIES_START} / {CAPABILITIES_END} to {path}."
        )
    new_text = pattern.sub(lambda _m: block, text)
    if new_text == text:
        return False
    path.write_text(new_text)
    return True


def _render_capabilities(registry: Registry) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="HeyNYC capabilities")
    table.add_column("Service", style="bold")
    table.add_column("What you can ask")
    table.add_column("Grounded in")
    table.add_column("Official link")
    for row in registry.capability_table():
        table.add_row(row.service, "\n".join(row.asks), row.grounded_in, row.official_link)
    Console().print(table)


def _cmd_capabilities(markdown: bool, write_readme: bool) -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    if write_readme:
        changed = _write_readme_capabilities(registry.capability_markdown())
        state = "updated" if changed else "already up to date"
        print(f"README capabilities section {state} → {_readme_path()}")
    elif markdown:
        print(registry.capability_markdown())
    else:
        _render_capabilities(registry)


def _cmd_modules() -> None:
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    if not registry.modules:
        print("No modules found in heynyc/modules/.")
        return
    print(f"{len(registry.modules)} module(s):")
    for module in registry.modules:
        print(f"  • {module.name} ({module.category}), {module.description}")
    print(f"\nDatasets: {list(registry.dataset_bindings())}")
    print(f"Allowlist: {registry.allowlist()}")


async def _cmd_index_build() -> None:
    from heynyc.core.index import default_embedder, open_store
    from heynyc.core.index.corpus import build_index

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    config.HEYNYC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    embedder = default_embedder()
    store = open_store(_INDEX_PATH, model_id=embedder.model_id)
    print(f"Building index for {len(registry.seeds())} seed URL(s)...")
    summary = await build_index(registry, store, embedder)
    print(f"  ok={summary['ok']}  chunks={summary['chunks']}  failed={len(summary['failed'])}")
    for fail in summary["failed"]:
        print(f"    ✗ {fail['url']}, {fail['error']}")


def _cmd_index_search(query: str, urls_only: bool = False) -> None:
    retriever = _load_retriever(required=True)
    if retriever is None:
        return
    for doc, score in retriever.search(query, k=5):
        if urls_only:
            print(doc.url)
        else:
            print(f"[{score:.2f}] {doc.title}, {doc.url}\n    {doc.text[:160]}...\n")


def _record_agent_turn(session_id: str, model: str, result) -> None:
    from heynyc.core import telemetry
    from heynyc.core.citations import used_citations

    used = used_citations(result.text, result.citations)

    telemetry.record_turn(
        telemetry.default_path(config.HEYNYC_DATA_DIR),
        session_id=session_id,
        model=model,
        usage=result.usage,
        n_tool_calls=len(result.tool_calls_made),
        tool_names=result.tool_calls_made,
        status=result.status,
        extra={
            "used_doc_citations": sum(
                citation.get("kind") == "DOC" for citation in used.values()
            )
        },
    )


async def _cmd_chat(question: str, model: str | None = None) -> None:
    # CLI --model wins when given; otherwise .env ALWAYS decides (owner rule 2026-07-21).
    from heynyc.core.pydantic_runtime import build_configured_runtime
    from heynyc.modules.advisories.tools import current_awareness

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    selected_model = model or config.HEYNYC_MODEL
    agent = build_configured_runtime(
        registry,
        model=selected_model,
        index=_load_retriever(required=False),
        current_awareness=current_awareness,
    )
    result = await agent.run(question, reminders=_default_reminders())
    print(result.text)
    from heynyc.core.citations import text_fragment_url, used_citations
    used = used_citations(result.text, result.citations)
    if used:
        print("\nSources:")
        for cid, c in used.items():
            url = text_fragment_url(c["url"], c.get("snippet", ""), c.get("kind", ""))
            print(f"  [{cid}] {c['title'] or c['url']} - {url}")
    _record_agent_turn("chat", selected_model, result)


def _render_stats(path) -> None:
    from rich.console import Console
    from rich.table import Table

    from heynyc.core import telemetry

    summary = telemetry.summarize(telemetry.load(path))
    console = Console()
    if not summary["turns"]:
        console.print(f"No telemetry yet at {path}. Run some `heynyc chat` turns first.")
        return
    table = Table(title=f"HeyNYC usage, {summary['turns']} turn(s)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total cost", f"${summary['total_cost_usd']:.4f}")
    table.add_row("cost / turn", f"${summary['cost_per_turn_usd']:.4f}")
    table.add_row("tokens in / out", f"{summary['input_tokens']} / {summary['output_tokens']}")
    table.add_row(
        "cached input tokens (total / scope)",
        f"{summary['cached_input_tokens']} / {summary.get('scope_cached_input_tokens', 0)}",
    )
    table.add_row("latency p50 / p95", f"{summary['latency_p50_ms']:.0f} / {summary['latency_p95_ms']:.0f} ms")
    table.add_row(
        "model / tool / other total",
        f"{summary['model_time_ms']:.0f} / {summary['tool_time_ms']:.0f} / "
        f"{summary['orchestration_time_ms']:.0f} ms",
    )
    table.add_row("model / tool calls", f"{summary['n_model_calls']} / {summary['n_tool_calls']}")
    table.add_row("iterations total", str(summary["iterations"]))
    table.add_row("error rate", f"{summary['error_rate'] * 100:.0f}%")
    table.add_row("tool mix", ", ".join(f"{k}×{v}" for k, v in summary["tool_mix"].items()) or "-")
    table.add_row(
        "index used / contributed / then web",
        f"{summary['index_search_turns']} / {summary['index_contributing_turns']} / "
        f"{summary['index_and_web_turns']}",
    )
    table.add_row("index contribution rate", f"{summary['index_contribution_rate'] * 100:.0f}%")
    table.add_row(
        "outcome mix",
        ", ".join(f"{k}×{v}" for k, v in summary["outcome_mix"].items()) or "-",
    )
    table.add_row(
        "checklist modules",
        ", ".join(f"{k}×{v}" for k, v in summary["scope_module_mix"].items()) or "-",
    )
    table.add_row(
        "checklist situations",
        ", ".join(f"{k}×{v}" for k, v in summary["scope_situation_mix"].items()) or "-",
    )
    console.print(table)


def _render_outcomes(telemetry_path, outcomes_path) -> None:
    """The find -> understand -> apply outcomes funnel: who reached the APPLY step, with
    drop-off. `turns` / `screened` / `apply started` come from telemetry `tool_names`; the
    two outcome stages come from the PII-free outcomes sidecar (see core/outcomes.py)."""
    from rich.console import Console
    from rich.table import Table

    from heynyc.core import outcomes, telemetry

    data = outcomes.funnel(telemetry.load(telemetry_path), outcomes.load(outcomes_path))
    counts = data["counts"]
    console = Console()
    if not counts["turns"]:
        console.print(f"No telemetry yet at {telemetry_path}. Run some turns first.")
        return
    labels = {
        "turns": "turns",
        "screened": "screened (screen_access_nyc_eligibility)",
        "eligible_shown": "eligible programs shown",
        "apply_started": "apply started (prepare_snap_application)",
        "form_ready": "filled form ready",
    }
    table = Table(title=f"HeyNYC outcomes funnel: {counts['turns']} turn(s)")
    table.add_column("stage")
    table.add_column("reached", justify="right")
    table.add_column("drop-off from prev", justify="right")
    for stage in outcomes.STAGES:
        drop = data["dropoff"].get(stage)
        drop_txt = "n/a" if drop is None else f"-{drop['lost']} ({drop['rate'] * 100:.0f}%)"
        table.add_row(labels[stage], str(counts[stage]), drop_txt)
    console.print(table)


def _feedback_path() -> Path:
    """Same log the channel orchestrator writes user-flagged turns into (build_deps)."""
    return config.HEYNYC_DATA_DIR / "feedback.jsonl"


def _sessions_dir() -> Path:
    return config.HEYNYC_DATA_DIR / "sessions"


def _channel_store():
    """Open the same channel store the server writes flag pointers into (read-only use here)."""
    from heynyc.channels.store import ChannelStore

    return ChannelStore(
        config.HEYNYC_DATA_DIR / "channels.sqlite3", rate_limit=config.CHANNEL_RATE_LIMIT,
        window_s=config.CHANNEL_RATE_WINDOW_S, dedup_ttl_s=config.CHANNEL_DEDUP_TTL_S,
    )


def _flagged_exchange(sessions_dir: Path, user_key: str, turn_index: int):
    """The one consented exchange a pointer references (the flagged assistant turn + the user turn
    before it), decrypted locally from the session JSONL exactly like the session tooling. None if
    the session was purged by retention or the pointer no longer resolves to an assistant turn."""
    from heynyc.core.session import _decode_line

    path = sessions_dir / f"{user_key}.jsonl"
    if not path.exists():
        return None
    turns: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        message = _decode_line(line)
        if message.get("_type") == "reset":
            turns.clear()
        elif message.get("role") in {"user", "assistant"}:
            turns.append(message)
    if not (0 <= turn_index < len(turns)) or turns[turn_index].get("role") != "assistant":
        return None
    user_turn = turns[turn_index - 1] if turn_index >= 1 and turns[turn_index - 1].get("role") == "user" else None
    return user_turn, turns[turn_index]


def _render_feedback(path, store=None, sessions_dir=None) -> None:
    from datetime import timezone

    from rich.console import Console
    from rich.table import Table

    from heynyc.channels import analytics

    console = Console()
    summary = analytics.summarize_feedback(analytics.load_feedback(path))
    if summary["total"]:
        console.print(f"[bold]HeyNYC feedback[/]: {summary['total']} flag(s) from {summary['users']} user(s)")
        console.print("flags: " + (", ".join(f"{k}×{v}" for k, v in summary["by_flag"].items()) or "-"))
        console.print("channels: " + (", ".join(f"{k}×{v}" for k, v in summary["by_channel"].items()) or "-"))

        if summary["top_queries"]:
            repeats = Table(title="repeat-flagged queries (a systematic error signal)")
            repeats.add_column("times", justify="right")
            repeats.add_column("flagged query")
            for query, count in summary["top_queries"]:
                if count > 1:
                    repeats.add_row(str(count), query)
            if repeats.row_count:
                console.print(repeats)

    # Triage view: each confirmed pointer joined to its session file and decrypted, showing exactly
    # the one exchange the resident consented to share (no surrounding context turns).
    pointers = store.flags() if store is not None else []
    if not summary["total"] and not pointers:
        console.print(f"No feedback yet at {path}. Residents flag a wrong answer with REPORT (then confirm).")
        return
    if not pointers:
        return
    console.print()
    console.print(f"[bold]Flagged exchanges for triage[/]: {len(pointers)} (decrypted locally, one exchange each)")
    for rec in pointers:
        when = datetime.fromtimestamp(rec["ts"], timezone.utc).isoformat()[:19] if rec.get("ts") else ""
        console.print(f"\n[dim]{when}[/]  flag=[bold]{rec.get('flag', '')}[/]  user={rec['user_key'][:12]}…")
        exchange = _flagged_exchange(sessions_dir, rec["user_key"], rec["turn_index"]) if sessions_dir else None
        if exchange is None:
            console.print("  [dim](session unavailable: purged by retention or pointer no longer resolves)[/]")
            continue
        user_turn, assistant_turn = exchange
        console.print(f"  [green]resident:[/] {(user_turn or {}).get('content', '') or '-'}")
        console.print(f"  [cyan]heynyc:[/] {assistant_turn.get('content', '')}")


# _append_segment / _reconcile_message_text / _approve_repl_action now live in
# heynyc.channels.console (the console channel owns the segment machinery + the local approver).
# The frozen `--raw` REPL below imports them back; the unified REPL rides ConsoleSink instead.


def _screen_turn_options(text: str) -> dict:
    from heynyc.channels.orchestrator import _SCREEN_REMINDER, _SCREEN_TOOL, is_screen

    requested = is_screen(text)
    return {
        "forced_tool": _SCREEN_TOOL if requested else None,
        "forced_tool_args": {
            "show_all": text.strip().lower() == "/screen all",
        } if requested else None,
        "excluded_tools": None if requested else {_SCREEN_TOOL},
        "screen_reminder": _SCREEN_REMINDER if requested else None,
    }


async def _cmd_repl(model: str | None = None, user: str = "local", temp: bool = False) -> None:
    """Interactive streaming chat that rides the SAME orchestrator path a texter does: the free
    commands (HELP / PRIVACY / REPORT / DELETE MY DATA), encrypted persistent sessions, identity,
    the per-resident spend cap, dedup, and channel rendering. Only the presentation differs, a
    transient live view (streamed text, tool notes, spinner) settling into the rendered markdown.
    `--user` keys a `console:<user>` identity (the seed of future account identity, NOT the OS user).
    Today's bare-agent REPL is frozen behind `--raw` (see `_cmd_repl_raw`)."""
    import uuid

    from rich.console import Console

    from heynyc.channels.base import InboundMessage
    from heynyc.channels.console import ConsoleReplier, build_console_deps
    from heynyc.channels.orchestrator import handle

    console = Console()
    temp_dir = None
    if temp:
        # Throwaway test session: random identity, isolated data dir, removed on exit. The full
        # channel machinery still runs (commands, welcome, caps), it just writes into a grave.
        import tempfile

        temp_dir = tempfile.TemporaryDirectory(prefix="heynyc-repl-temp-")
        user = f"temp-{uuid.uuid4().hex[:8]}"
    deps = build_console_deps(
        console=console, model=model,
        data_dir=Path(temp_dir.name) if temp_dir else None,
    )
    sink = deps.event_sink
    artifacts_dir = (Path(temp_dir.name) if temp_dir else config.HEYNYC_DATA_DIR) / "repl-artifacts"
    replier = ConsoleReplier(console, artifacts_dir)

    modules = ", ".join(m.name for m in deps.agent.registry.modules)
    console.print(
        "[bold]HeyNYC[/], ask about NYC services & events. [dim]Ctrl-C to exit.[/]\n"
        "[dim]Commands (all work here, same as texting): HELP, PRIVACY, REPORT, DELETE MY DATA.[/]"
    )
    console.print(f"[dim]Modules loaded: {modules or 'none'}  ·  you = console:{user}[/]\n")

    while True:
        try:
            question = console.input("[bold green]you ▸ [/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return
        if not question:
            continue
        # A fresh message id per turn: dedup keys on it, so an identical repeated question is not
        # silently swallowed as a duplicate.
        inbound = InboundMessage(
            channel="console", sender=user, text=question, message_id=str(uuid.uuid4()),
        )
        console.print("[bold cyan]heynyc ▸[/]")
        sink.start_turn()
        try:
            await handle(inbound, replier, deps)
        finally:
            sink.finish()
        console.print()
    if temp_dir is not None:
        temp_dir.cleanup()


async def _cmd_repl_raw(model: str | None = None) -> None:
    """The frozen legacy REPL, kept verbatim behind `--raw` as a DEBUG surface: a bare Agent that
    bypasses the channel features (free commands, encrypted sessions, identity, spend cap, channel
    rendering). The default `repl` now rides the shared orchestrator path, see `_cmd_repl`."""
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.spinner import Spinner
    from rich.text import Text

    from heynyc.channels.console import (
        _append_segment,
        _approve_repl_action,
        _reconcile_message_text,
    )

    console = Console()
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)

    async def approver(name: str, args: dict) -> bool:
        return _approve_repl_action(console, name, args)

    from heynyc.modules.advisories.tools import current_awareness

    agent = Agent(
        registry, model=model or config.HEYNYC_MODEL, index=_load_retriever(required=False),
        approver=approver, notify_awareness=current_awareness, scope_gate=True,
    )
    convo = agent.conversation()

    # Form-fill test surface: persist a structured draft + collect generated PDFs, so the REPL
    # exercises the full pipeline. (Set HEYNYC_FORMS=true to expose the SNAP application tool.)
    from heynyc.core.drafts import DraftStore

    drafts = DraftStore(config.HEYNYC_DATA_DIR / "repl-drafts").for_user("repl-user")
    artifacts_dir = config.HEYNYC_DATA_DIR / "repl-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold]HeyNYC[/] [yellow](--raw debug surface)[/]: bare agent, no channel commands, "
        "sessions, identity, or rendering. [dim]Ctrl-C to exit.[/]\n"
    )
    modules = ", ".join(m.name for m in registry.modules)
    console.print(f"[dim]Modules loaded: {modules or 'none'}[/]\n")

    while True:
        try:
            question = console.input("[bold green]you ▸ [/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return
        if not question:
            continue

        segments: list = []  # ordered render parts: {"kind": "text"|"tool", "text": str}
        citations: dict = {}
        done = False
        turn_result = None
        message_start = 0

        def render():
            parts: list = []
            for seg in segments:
                if seg["kind"] == "tool":
                    parts.append(Text(seg["text"], style="dim cyan"))
                elif seg["text"].strip():
                    parts.append(Markdown(seg["text"]))
            # Spinner while we wait (before any output, or after a tool note until the
            # answer starts streaming), hidden once answer text is flowing or we're done.
            answering = bool(segments) and segments[-1]["kind"] == "text" and segments[-1]["text"].strip()
            if not done and not answering:
                parts.append(Spinner("dots", text=Text(" thinking…", style="dim")))
            return Group(*parts)

        before_pdfs = set(artifacts_dir.glob("*.pdf"))
        screen_options = _screen_turn_options(question)
        reminders = _default_reminders()
        if screen_options["screen_reminder"]:
            reminders.append(screen_options["screen_reminder"])
        console.print("[bold cyan]heynyc ▸[/]")
        with Live(render(), console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            async for event in convo.stream(
                question,
                reminders=reminders,
                output_dir=artifacts_dir,
                drafts=drafts,
                forced_tool=screen_options["forced_tool"],
                forced_tool_args=screen_options["forced_tool_args"],
                excluded_tools=screen_options["excluded_tools"],
            ):
                if isinstance(event, events.MessageStart):
                    message_start = len(segments)
                elif isinstance(event, events.ToolStart):
                    _append_segment(segments, "tool", f"· using {event.name}…")
                elif isinstance(event, events.TextDelta):
                    _append_segment(segments, "text", event.text)
                elif isinstance(event, events.MessageCompleted):
                    _reconcile_message_text(segments, message_start, event.text)
                elif isinstance(event, events.Done):
                    citations = event.citations
                    turn_result = event.result
                    done = True
                live.update(render())

        if turn_result is not None:
            _record_agent_turn("repl", agent.model, turn_result)

        from heynyc.core.citations import text_fragment_url, used_citations
        answer_text = "".join(s["text"] for s in segments if s["kind"] == "text")
        used = used_citations(answer_text, citations)
        if used:
            console.print("[dim]Sources:[/]")
            for cid, c in used.items():
                url = text_fragment_url(c["url"], c.get("snippet", ""), c.get("kind", ""))
                console.print(f"  [dim]\\[{cid}] {c['title'] or c['url']} - {url}[/]")
        for pdf in sorted(set(artifacts_dir.glob("*.pdf")) - before_pdfs):
            console.print(f"[bold green]📄 saved your filled draft →[/] {pdf}")
        console.print()


# Conservative per-case live-cost planning figure for the eval cost guard, from observed
# GPT-5.4-mini acceptance runs (simple denials ~$0.0002, tool-heavy events ~$0.02).
_EVAL_COST_PER_CASE_USD = 0.02


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _repeat_eval_cases(cases: list, explicit_case_ids: list[str]) -> list:
    """Repeat explicit case selections; otherwise keep the broad-run safety subset."""
    if explicit_case_ids:
        selected = set(explicit_case_ids)
        return [case for case in cases if case.id in selected]
    return [case for case in cases if case.safety_critical]


def _eval_run_metadata(
    model: str,
    results: list,
    *,
    repeat_summary: dict | None = None,
) -> dict:
    from heynyc.eval.bench import _candidate_cost

    usage_results = [
        turn
        for result in results
        for turn in (getattr(result, "turn_results", None) or [result])
    ]
    cost, input_tokens, output_tokens = _candidate_cost(model, results)
    metadata = {
        "model": model,
        "case_ids": list(dict.fromkeys(result.case.id for result in results)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "candidate_cost_usd": cost,
        "latency_ms": sum(float(result.usage.get("latency_ms", 0) or 0) for result in usage_results),
        "n_model_calls": sum(int(result.usage.get("n_model_calls", 0) or 0) for result in usage_results),
        "n_tool_calls": sum(int(result.usage.get("n_tool_calls", 0) or 0) for result in usage_results),
    }
    if repeat_summary is not None:
        metadata["repeat"] = repeat_summary
    return metadata


def _live_eval_sink(directory: Path, case_id: str):
    """Persist the native event stream and print a compact live trajectory."""
    import time

    from heynyc.eval.report import event_writer

    write = event_writer(directory, case_id)
    started = time.monotonic()
    tool_calls = 0
    print(f"\n[{case_id}] started")

    def on_event(event) -> None:
        nonlocal tool_calls
        write(event)
        elapsed = time.monotonic() - started
        if isinstance(event, events.ToolStart):
            tool_calls += 1
            print(f"[{case_id}] {elapsed:6.1f}s  tool {tool_calls}: {event.name}", flush=True)
        elif isinstance(event, events.ToolCompleted):
            summary = f"  {event.result_summary}" if event.result_summary else ""
            print(f"[{case_id}] {elapsed:6.1f}s  {event.name}: {event.status}{summary}", flush=True)
        elif isinstance(event, events.ErrorEvent):
            print(f"[{case_id}] {elapsed:6.1f}s  error: {event.message}", flush=True)
        elif isinstance(event, events.Done):
            usage = getattr(event.result, "usage", {}) or {}
            calls = int(usage.get("n_model_calls", 0) or usage.get("requests", 0) or 0)
            tokens = int(usage.get("input_tokens", 0) or 0) + int(
                usage.get("output_tokens", 0) or 0
            )
            cost = usage.get("cost_usd")
            cost_text = f"${cost:.5f}" if isinstance(cost, (int, float)) else "unpriced"
            print(
                f"[{case_id}] {elapsed:6.1f}s  done: {calls} model calls, "
                f"{tool_calls} tool calls, {tokens} tokens, {cost_text}",
                flush=True,
            )

    return on_event


async def _cmd_eval(
    use_api_judge: bool,
    repeat: int = 1,
    out: str | None = None,
    module: str | None = None,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
    run_all_cases: bool = False,
    model: str | None = None,
) -> None:
    from datetime import timezone
    from pathlib import Path

    from heynyc.eval import evaluate, load_cases, run_all, run_repeated, write_run
    from heynyc.eval.bench import build_eval_agent
    from heynyc.eval.cases import select_cases
    from heynyc.eval.report import progress_writer

    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    cases = load_cases(registry)
    unselective = not (module or case_ids or tags or sample)
    if unselective and not run_all_cases:
        estimate = len(cases) * _EVAL_COST_PER_CASE_USD
        print(
            f"This would run all {len(cases)} eval cases live (roughly ${estimate:.2f} at "
            f"planning rates). Full runs are for large changes: pass --all to confirm, or "
            f"select with --module, --case, --tag, or --sample."
        )
        return
    retriever = _load_retriever(required=False)
    cases = select_cases(cases, module=module, case_ids=case_ids or None, tags=tags or None,
                         sample=sample, seed=seed)
    if not cases:
        scope = f" for module '{module}'" if module else ""
        print(f"No eval cases found{scope} (modules need an eval.yaml).")
        return
    print(f"Running {len(cases)} eval case(s) across {len(registry.modules)} module(s)...")

    selected_model = model or config.HEYNYC_MODEL

    def factory():
        return build_eval_agent(registry, selected_model, retriever)

    run_dir = Path(out) if out else (
        config.HEYNYC_DATA_DIR / "eval" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    )
    results = await run_all(
        factory,
        cases,
        reminders=_default_reminders(),
        on_case=progress_writer(run_dir),
        event_sink_factory=lambda case: _live_eval_sink(run_dir, case.id),
    )
    judge = None
    if use_api_judge:
        # The PAID, opt-in API judge. Thread today's date through so it treats live/future-dated
        # tool data as current rather than "outdated". The default judge is the interactive Agent
        # reviewing the traces (free), it needs no in-harness call.
        from heynyc.eval.judges import make_api_judge

        judge = make_api_judge(config.HEYNYC_JUDGE_MODEL, now=datetime.now())
    report = await evaluate(results, judge=judge)
    print("\n" + report.render())

    # pass^k reliability on the safety-critical subset (customer-facing metric).
    repeat_summary = None
    repeated_artifacts = []
    billed_results = list(results)
    repeat_gate_passed = True
    if repeat > 1:
        from heynyc.eval.report import GateReport

        repeat_targets = _repeat_eval_cases(cases, case_ids or [])
        reliable = 0
        repeated_cases = []
        for case in repeat_targets:
            initial = next(result for result in results if result.case.id == case.id)
            runs = [initial]
            runs.extend(
                await run_repeated(
                    factory,
                    case,
                    k=repeat - 1,
                    reminders=_default_reminders(),
                    event_sink_factory=lambda repeated_case, index: _live_eval_sink(
                        run_dir / "repeats" / repeated_case.id / f"run-{index + 2:02d}",
                        repeated_case.id,
                    ),
                )
            )
            billed_results.extend(runs[1:])
            sub = await evaluate(runs)
            outcomes = [case_report.passed for case_report in sub.reports]
            is_reliable = all(outcomes)
            repeat_gate_passed = repeat_gate_passed and is_reliable
            if is_reliable:
                reliable += 1
            artifact_paths = [
                f"repeats/{case.id}/run-{index:02d}"
                for index in range(1, len(runs) + 1)
            ]
            repeated_artifacts.append(
                (
                    case.id,
                    runs,
                    [GateReport(reports=[case_report]) for case_report in sub.reports],
                )
            )
            repeated_cases.append({
                "case_id": case.id,
                "passed": outcomes,
                "reliable": is_reliable,
                "artifacts": artifact_paths,
            })
        if repeat_targets:
            repeat_summary = {
                "k": repeat,
                "eligible_case_count": len(repeat_targets),
                "reliable_case_count": reliable,
                "cases": repeated_cases,
            }
            label = "explicit cases" if case_ids else "safety subset"
            print(
                f"\npass^{repeat} ({label}): {reliable}/{len(repeat_targets)} "
                f"cases reliable across {repeat} runs"
            )

    for case_id, runs, reports in repeated_artifacts:
        for index, (run, run_report) in enumerate(zip(runs, reports), start=1):
            write_run(
                run_dir / "repeats" / case_id / f"run-{index:02d}",
                run_report,
                metadata=_eval_run_metadata(selected_model, [run]),
            )
    write_run(
        run_dir,
        report,
        metadata=_eval_run_metadata(
            selected_model,
            billed_results,
            repeat_summary=repeat_summary,
        ),
        overall_passed=report.promotion_ready and repeat_gate_passed,
    )
    print(f"\nRun written to {run_dir}")
    raise SystemExit(0 if report.promotion_ready and repeat_gate_passed else 1)


async def _cmd_bench(models: list[str], module: str | None, use_api_judge: bool, out: str | None) -> None:
    """Run the golden eval cases across several candidate models and print a per-model comparison."""
    from heynyc.eval import load_cases, render_bench, run_bench

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    retriever = _load_retriever(required=False)
    cases = load_cases(registry)
    if module:
        cases = [c for c in cases if c.module == module]
    if not cases:
        scope = f" for module '{module}'" if module else ""
        print(f"No eval cases found{scope} (modules need an eval.yaml).")
        return
    safety_ids = {c.id for c in cases if c.safety_critical}
    print(f"Benching {len(models)} model(s) on {len(cases)} case(s): {', '.join(models)}")

    judge = None
    if use_api_judge:
        # The PAID, opt-in API judge, same one `eval` uses. Thread today's date so live/future-dated
        # tool data reads as current, not "outdated".
        from heynyc.eval.judges import make_api_judge

        judge = make_api_judge(config.HEYNYC_JUDGE_MODEL, now=datetime.now())

    rows = await run_bench(
        models, registry, retriever, cases, reminders=_default_reminders(), judge=judge, out_dir=out,
    )
    print("\n" + render_bench(rows, safety_ids))


def _run_repl(model=None, raw: bool = False, user: str = "local", temp: bool = False) -> None:
    try:
        asyncio.run(_cmd_repl_raw(model) if raw else _cmd_repl(model, user, temp))
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="heynyc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("modules", help="list discovered service modules")
    newmod = sub.add_parser("new-module", help="scaffold a new service module")
    newmod.add_argument("name")
    sub.add_parser("index-build", help="fetch + embed module seeds into the index")
    isearch = sub.add_parser("index-search", help="query the index directly (no LLM)")
    isearch.add_argument("--urls-only", action="store_true", help="print one result URL per line")
    isearch.add_argument("query")
    chat = sub.add_parser("chat", help="ask the agent a question (one-shot)")
    chat.add_argument("--model", default=None, help="answer model override; without it, .env (HEYNYC_MODEL) ALWAYS decides")
    chat.add_argument("question")
    repl = sub.add_parser("repl", help="interactive streaming chat on the same path as texting")
    repl.add_argument("--model", default=None, help="answer model override; without it, .env (HEYNYC_MODEL) ALWAYS decides")
    repl.add_argument("--temp", action="store_true",
                      help="throwaway session: a random identity in a temporary data dir, nothing persists after exit")
    repl.add_argument("--user", default="local",
                      help="identity for this session, keyed as console:<user> (NOT your OS username); "
                           "the seed of future account identity")
    repl.add_argument("--raw", action="store_true",
                      help="legacy bare-agent REPL: a DEBUG surface that bypasses channel commands, "
                           "encrypted sessions, identity, spend cap, and channel rendering")
    cap = sub.add_parser("capabilities", help="print the capabilities table (generated from module manifests)")
    cap.add_argument("--markdown", action="store_true", help="emit a GitHub markdown table")
    cap.add_argument("--write-readme", dest="write_readme", action="store_true",
                     help="rewrite the CAPABILITIES section of README.md in place (idempotent)")
    sub.add_parser("stats", help="show cost/usage telemetry from past chat turns")
    sub.add_parser("outcomes", help="show the find->understand->apply funnel (who reached APPLY)")
    sub.add_parser("feedback", help="review user-flagged wrong answers (the error-feedback loop)")
    ev = sub.add_parser("eval", help="run the no-hallucination eval gate")
    ev.add_argument("--model", default=None, help="answer model override; without it, .env (HEYNYC_MODEL) ALWAYS decides")
    ev.add_argument(
        "--api-judge", dest="api_judge", action="store_true",
        help="also run the PAID API groundedness judge (a cross-family LLM call) for "
             "parity/reproducibility. The default internal judge is the interactive Agent "
             "reviewing the run traces (no API cost).",
    )
    # Back-compat hidden alias: `--judge` still maps to the same PAID API judge.
    ev.add_argument("--judge", dest="api_judge", action="store_true", help=argparse.SUPPRESS)
    ev.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="repeat explicit --case selections K times, or the safety subset on broader runs",
    )
    ev.add_argument("--out", default=None, help="run directory to write traces + report into")
    ev.add_argument("--module", default=None, help="only run cases from this module (e.g. benefits)")
    ev.add_argument("--case", dest="case_ids", action="append", default=[],
                    help="only run this case id; repeat for more than one")
    ev.add_argument("--tag", dest="tags", action="append", default=[],
                    help="only run cases carrying this tag (e.g. a failure-db id like F046)")
    ev.add_argument("--sample", type=int, default=None,
                    help="run a deterministic random sample of N cases (after other filters)")
    ev.add_argument("--seed", type=int, default=0, help="seed for --sample (default 0)")
    ev.add_argument("--list", dest="list_cases", action="store_true",
                    help="print every case on one line (id, source, flags, tags) and exit")
    ev.add_argument("--all", dest="run_all_cases", action="store_true",
                    help="confirm running the FULL live case set (large changes only)")
    bench = sub.add_parser("bench", help="run the eval cases across several candidate models and compare")
    bench.add_argument("--models", required=True,
                       help="comma-separated model ids to bench, e.g. gpt-5,claude-sonnet-4,gemini-2")
    bench.add_argument("--module", default=None, help="only run cases from this module (e.g. benefits)")
    bench.add_argument("--api-judge", dest="api_judge", action="store_true",
                       help="also run the PAID API groundedness judge for each model (see `eval --api-judge`)")
    bench.add_argument("--out", default=None, help="directory to write each model's traces + report into (one subdir per model)")
    serve = sub.add_parser("serve", help="run the messaging webhook server (WhatsApp/SMS)")
    serve.add_argument("--provider", default=None, help="meta | twilio | both (default: env WHATSAPP_PROVIDER)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "modules":
        _cmd_modules()
    elif args.command == "new-module":
        _cmd_new_module(args.name)
    elif args.command == "index-build":
        asyncio.run(_cmd_index_build())
    elif args.command == "index-search":
        _cmd_index_search(args.query, urls_only=args.urls_only)
    elif args.command == "chat":
        asyncio.run(_cmd_chat(args.question, model=args.model))
    elif args.command == "repl":
        _run_repl(getattr(args, "model", None), raw=args.raw, user=args.user, temp=args.temp)
    elif args.command == "capabilities":
        _cmd_capabilities(markdown=args.markdown, write_readme=args.write_readme)
    elif args.command == "stats":
        from heynyc.core import telemetry

        _render_stats(telemetry.default_path(config.HEYNYC_DATA_DIR))
    elif args.command == "outcomes":
        from heynyc.core import outcomes, telemetry

        _render_outcomes(telemetry.default_path(config.HEYNYC_DATA_DIR),
                         outcomes.default_path(config.HEYNYC_DATA_DIR))
    elif args.command == "feedback":
        _render_feedback(_feedback_path(), store=_channel_store(), sessions_dir=_sessions_dir())
    elif args.command == "eval":
        if args.list_cases:
            from heynyc.eval.cases import render_case_listing
            registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
            print(render_case_listing(registry))
            return
        asyncio.run(_cmd_eval(model=args.model, use_api_judge=args.api_judge, repeat=args.repeat, out=args.out,
                              module=args.module, case_ids=args.case_ids,
                              tags=args.tags, sample=args.sample, seed=args.seed,
                              run_all_cases=args.run_all_cases))
    elif args.command == "bench":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        asyncio.run(_cmd_bench(models=models, module=args.module, use_api_judge=args.api_judge, out=args.out))
    elif args.command == "serve":
        import uvicorn

        from heynyc.channels.app import create_app

        uvicorn.run(create_app(provider=args.provider), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
