"""HeyNYC CLI.

    uv run python -m heynyc modules           # list discovered service modules
    uv run python -m heynyc index-build        # fetch + embed module seeds into the index
    uv run python -m heynyc index-search "q"   # query the index (no LLM)
    uv run python -m heynyc chat "question"    # ask the agent (needs an LLM key)
"""
from __future__ import annotations

import argparse
import asyncio
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
    store = open_store(_INDEX_PATH)
    return IndexRetriever(store=store, embedder=default_embedder())


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
    store = open_store(_INDEX_PATH)
    print(f"Building index for {len(registry.seeds())} seed URL(s)...")
    summary = await build_index(registry, store, default_embedder())
    print(f"  ok={summary['ok']}  chunks={summary['chunks']}  failed={len(summary['failed'])}")
    for fail in summary["failed"]:
        print(f"    ✗ {fail['url']}, {fail['error']}")


def _cmd_index_search(query: str) -> None:
    retriever = _load_retriever(required=True)
    if retriever is None:
        return
    for doc, score in retriever.search(query, k=5):
        print(f"[{score:.2f}] {doc.title}, {doc.url}\n    {doc.text[:160]}...\n")


async def _cmd_chat(question: str) -> None:
    from heynyc.core import telemetry

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    agent = Agent(registry, model=config.HEYNYC_MODEL, index=_load_retriever(required=False))
    result = await agent.run(question, reminders=_default_reminders())
    print(result.text)
    from heynyc.core.citations import text_fragment_url, used_citations
    used = used_citations(result.text, result.citations)
    if used:
        print("\nSources:")
        for cid, c in used.items():
            url = text_fragment_url(c["url"], c.get("snippet", ""), c.get("kind", ""))
            print(f"  [{cid}] {c['title'] or c['url']} - {url}")
    telemetry.record_turn(
        telemetry.default_path(config.HEYNYC_DATA_DIR), session_id="chat", model=agent.model,
        usage=result.usage, n_tool_calls=len(result.tool_calls_made),
        tool_names=result.tool_calls_made, status=result.status,
    )


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
        "screened": "screened (screen_eligibility)",
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


def _render_feedback(path) -> None:
    from rich.console import Console
    from rich.table import Table

    from heynyc.channels import analytics

    summary = analytics.summarize_feedback(analytics.load_feedback(path))
    console = Console()
    if not summary["total"]:
        console.print(f"No feedback yet at {path}. Residents flag a wrong answer with `wrong` / `/wrong <note>`.")
        return
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

    recent = Table(title="most recent flags")
    recent.add_column("when", style="dim")
    recent.add_column("flag")
    recent.add_column("flagged query")
    recent.add_column("reason (redacted)")
    for rec in summary["recent"]:
        recent.add_row(rec.get("ts", "")[:19], rec.get("flag", ""),
                       rec.get("user_query", "") or "-", rec.get("note", "") or "-")
    console.print(recent)


def _append_segment(segments: list, kind: str, text: str) -> None:
    """Accumulate a stream event into ordered REPL render segments.

    Consecutive text deltas merge into one block; a tool note breaks the text so
    ordering is preserved, a tool call that arrives after some preamble text renders
    *below* it (a chronological stack), not pinned above the whole message."""
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


async def _cmd_repl() -> None:
    """Interactive, streaming multi-turn chat, rich-rendered, Claude-Code-like."""
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.spinner import Spinner
    from rich.text import Text

    console = Console()
    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)

    async def approver(name: str, args: dict) -> bool:
        return _approve_repl_action(console, name, args)

    agent = Agent(
        registry, model=config.HEYNYC_MODEL, index=_load_retriever(required=False), approver=approver
    )
    convo = agent.conversation()

    # Form-fill test surface: persist a structured draft + collect generated PDFs, so the REPL
    # exercises the full pipeline. (Set HEYNYC_FORMS=true to expose the SNAP application tool.)
    from heynyc.core.drafts import DraftStore

    drafts = DraftStore(config.HEYNYC_DATA_DIR / "repl-drafts").for_user("repl-user")
    artifacts_dir = config.HEYNYC_DATA_DIR / "repl-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]HeyNYC[/], ask about NYC services & events. [dim]Ctrl-C to exit.[/]\n")
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
        console.print("[bold cyan]heynyc ▸[/]")
        with Live(render(), console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            async for event in convo.stream(question, reminders=_default_reminders(),
                                             output_dir=artifacts_dir, drafts=drafts):
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
                    done = True
                live.update(render())

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


async def _cmd_eval(use_api_judge: bool, repeat: int = 1, out: str | None = None, module: str | None = None) -> None:
    from datetime import timezone
    from pathlib import Path

    from heynyc.core.agent import Agent
    from heynyc.eval import evaluate, load_cases, run_all, run_repeated, write_run

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST, config.NEWS_ALLOWLIST)
    retriever = _load_retriever(required=False)
    cases = load_cases(registry)
    if module:
        cases = [c for c in cases if c.module == module]
    if not cases:
        scope = f" for module '{module}'" if module else ""
        print(f"No eval cases found{scope} (modules need an eval.yaml).")
        return
    print(f"Running {len(cases)} eval case(s) across {len(registry.modules)} module(s)...")

    def factory():
        return Agent(registry, model=config.HEYNYC_MODEL, index=retriever)

    results = await run_all(factory, cases, reminders=_default_reminders())
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
    if repeat > 1:
        safety = [c for c in cases if c.safety_critical]
        reliable = 0
        for case in safety:
            runs = await run_repeated(factory, case, k=repeat, reminders=_default_reminders())
            sub = await evaluate(runs)
            if all(r.passed for r in sub.reports):
                reliable += 1
        if safety:
            print(f"\npass^{repeat} (safety subset): {reliable}/{len(safety)} cases reliable across {repeat} runs")

    run_dir = Path(out) if out else (
        config.HEYNYC_DATA_DIR / "eval" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    )
    write_run(run_dir, report)
    print(f"\nRun written to {run_dir}")
    raise SystemExit(0 if report.passed else 1)


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="heynyc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("modules", help="list discovered service modules")
    newmod = sub.add_parser("new-module", help="scaffold a new service module")
    newmod.add_argument("name")
    sub.add_parser("index-build", help="fetch + embed module seeds into the index")
    isearch = sub.add_parser("index-search", help="query the index directly (no LLM)")
    isearch.add_argument("query")
    chat = sub.add_parser("chat", help="ask the agent a question (one-shot)")
    chat.add_argument("question")
    sub.add_parser("repl", help="interactive streaming chat (feels like Claude Code)")
    cap = sub.add_parser("capabilities", help="print the capabilities table (generated from module manifests)")
    cap.add_argument("--markdown", action="store_true", help="emit a GitHub markdown table")
    cap.add_argument("--write-readme", dest="write_readme", action="store_true",
                     help="rewrite the CAPABILITIES section of README.md in place (idempotent)")
    sub.add_parser("stats", help="show cost/usage telemetry from past chat turns")
    sub.add_parser("outcomes", help="show the find->understand->apply funnel (who reached APPLY)")
    sub.add_parser("feedback", help="review user-flagged wrong answers (the error-feedback loop)")
    ev = sub.add_parser("eval", help="run the no-hallucination eval gate")
    ev.add_argument(
        "--api-judge", dest="api_judge", action="store_true",
        help="also run the PAID API groundedness judge (a cross-family LLM call) for "
             "parity/reproducibility. The default internal judge is the interactive Agent "
             "reviewing the run traces (no API cost).",
    )
    # Back-compat hidden alias: `--judge` still maps to the same PAID API judge.
    ev.add_argument("--judge", dest="api_judge", action="store_true", help=argparse.SUPPRESS)
    ev.add_argument("--repeat", type=int, default=1, help="run the safety subset K times and report pass^K")
    ev.add_argument("--out", default=None, help="run directory to write traces + report into")
    ev.add_argument("--module", default=None, help="only run cases from this module (e.g. benefits)")
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
        _cmd_index_search(args.query)
    elif args.command == "chat":
        asyncio.run(_cmd_chat(args.question))
    elif args.command == "repl":
        asyncio.run(_cmd_repl())
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
        _render_feedback(_feedback_path())
    elif args.command == "eval":
        asyncio.run(_cmd_eval(use_api_judge=args.api_judge, repeat=args.repeat, out=args.out, module=args.module))
    elif args.command == "bench":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        asyncio.run(_cmd_bench(models=models, module=args.module, use_api_judge=args.api_judge, out=args.out))
    elif args.command == "serve":
        import uvicorn

        from heynyc.channels.app import create_app

        uvicorn.run(create_app(provider=args.provider), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
