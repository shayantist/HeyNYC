"""One conversational turn against a persistent session, for simulated resident testing.

Rides the SAME orchestrator path a texter uses (`build_console_deps` + `orchestrator.handle`), so
sessions, the welcome, free commands, spend caps, dedup, and channel rendering all behave as in
production. It reinvents nothing: `repl` drives the identical machinery interactively, and this is
the non-interactive, one-turn-at-a-time form a persona agent can script.

Why this exists: the eval harness plays multi-turn cases, but its turns are fixed in YAML. A
simulated resident has to READ the reply and decide what to say next, which static cases cannot do.

    uv run python scripts/persona_turn.py --user maria-es "necesito comida hoy"

Prints one JSON object: the reply text, the session key, and turn metadata. Keep passing the same
`--user` to continue a conversation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console


class _CapturingReplier:
    """Collects what the resident would receive instead of rendering it."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.documents: list[str] = []

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_document(self, path: str, caption: str = "") -> None:
        self.documents.append(f"{path} :: {caption}".strip(" :"))

    async def indicate_typing(self) -> None:
        return None


def _trace_result(result: object | None) -> dict:
    if result is None:
        return {}
    return {
        key: getattr(result, key)
        for key in (
            "text",
            "status",
            "citations",
            "tool_calls_made",
            "iterations",
            "messages",
            "usage",
            "diagnostics",
        )
    }


async def _run(
    user: str,
    text: str,
    data_dir: Path | None,
    trace_dir: Path | None = None,
) -> dict:
    from heynyc.channels.base import InboundMessage
    from heynyc.channels.console import build_console_deps
    from heynyc.channels.orchestrator import handle
    from heynyc.core.events import Done

    console = Console(quiet=True, file=open("/dev/null", "w"))
    deps = build_console_deps(console=console, data_dir=data_dir)
    index_enabled = "index_search" in getattr(getattr(deps, "agent", None), "tools", {})
    replier = _CapturingReplier()
    events: list[dict] = []
    result = None

    def capture(event: object) -> None:
        nonlocal result
        events.append({"type": event.type, **event.sse_data()})
        if isinstance(event, Done):
            result = event.result

    deps.event_sink = capture
    inbound = InboundMessage(
        channel="console", sender=user, text=text, message_id=str(uuid.uuid4()),
    )
    error = None
    try:
        await handle(inbound, replier, deps)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    output = {
        "user": user,
        "sent": text,
        "reply": "\n\n".join(replier.texts),
        "documents": replier.documents,
        "reply_count": len(replier.texts),
        "index_enabled": index_enabled,
        "events": events,
        "agent_result": _trace_result(result),
    }
    if error is not None:
        output["error"] = error
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{uuid.uuid4()}.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        output["trace_path"] = str(path)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="what the resident says this turn")
    parser.add_argument("--user", required=True, help="stable persona id; same id continues the thread")
    parser.add_argument("--data-dir", default=None, help="isolate a persona's state from the default")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="save the complete observable turn trace here; defaults to DATA_DIR/traces",
    )
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    data_dir = Path(args.data_dir) if args.data_dir else None
    trace_dir = Path(args.trace_dir) if args.trace_dir else data_dir / "traces" if data_dir else None
    result = asyncio.run(_run(args.user, args.text, data_dir, trace_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int("error" in result)


if __name__ == "__main__":
    sys.exit(main())
