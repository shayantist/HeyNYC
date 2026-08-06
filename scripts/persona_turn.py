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

from rich.console import Console

from heynyc.channels.base import InboundMessage
from heynyc.channels.console import build_console_deps
from heynyc.channels.orchestrator import handle


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


async def _run(user: str, text: str, data_dir: Path | None) -> dict:
    console = Console(quiet=True, file=open("/dev/null", "w"))
    deps = build_console_deps(console=console, data_dir=data_dir)
    replier = _CapturingReplier()
    inbound = InboundMessage(
        channel="console", sender=user, text=text, message_id=str(uuid.uuid4()),
    )
    await handle(inbound, replier, deps)
    return {
        "user": user,
        "sent": text,
        "reply": "\n\n".join(replier.texts),
        "documents": replier.documents,
        "reply_count": len(replier.texts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="what the resident says this turn")
    parser.add_argument("--user", required=True, help="stable persona id; same id continues the thread")
    parser.add_argument("--data-dir", default=None, help="isolate a persona's state from the default")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else None
    result = asyncio.run(_run(args.user, args.text, data_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
