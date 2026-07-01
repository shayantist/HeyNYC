"""The universal, channel-agnostic flow: dedup -> rate-limit -> per-user lock ->
bounded concurrency -> run the agent on the user's session -> reply -> record.
Flag keywords short-circuit into the feedback log instead of the agent."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

from heynyc.core.drafts import DraftStore
from datetime import datetime, timezone
from pathlib import Path

from heynyc.core.agent import Agent
from heynyc.core.session import Session

from . import analytics
from .base import InboundMessage, KeyedLocks, Replier
from .format import render
from .identity import user_key
from .store import ChannelStore

_FLAG_TOKENS = {"wrong", "report", "incorrect", "bad answer", "👎"}
_HELP_TOKENS = {"hi", "hello", "hey", "help", "menu", "start", "/help", "/menu",
                "what can you do", "what can i ask", "what do you do"}
_RATE_LIMIT_MSG = "You're sending a lot at once — give me a moment and try again shortly. 🙏"


@dataclass
class Deps:
    agent: Agent
    store: ChannelStore
    sessions_dir: Path
    salt: str
    telemetry_path: Path
    feedback_path: Path
    locks: KeyedLocks
    semaphore: asyncio.Semaphore
    drafts: Optional[DraftStore] = None   # per-user structured form drafts (None → no persistence)


def is_flag(text: str) -> bool:
    return text.strip().lower() in _FLAG_TOKENS


def is_help(text: str) -> bool:
    """A greeting / 'what can you do' — answered with the grounded capability menu, not the agent."""
    return text.strip().lower().rstrip("!?. ") in _HELP_TOKENS


def _reminders() -> list[str]:
    return [f"Today's date is {datetime.now():%A, %B %-d, %Y}. The user is in New York City."]


def _last(turns: list[dict], role: str) -> str:
    for turn in reversed(turns):
        if turn.get("role") == role:
            return turn.get("content") or ""
    return ""


def _artifacts_in(art_dir: Path) -> list[str]:
    """Files a tool wrote into THIS request's artifacts dir. The orchestrator owns this directory,
    so untrusted tool-result content can never influence which file is read/sent — we list the dir,
    we never parse a path out of model/tool text (that would be an arbitrary-file-read sink)."""
    return sorted(str(p) for p in art_dir.glob("*") if p.is_file())


async def handle(msg: InboundMessage, replier: Replier, deps: Deps) -> None:
    if deps.store.seen(msg.message_id):       # dedup (also records) — before any work
        return
    key = user_key(msg.channel, msg.sender, deps.salt)
    if not deps.store.allow(key):
        await replier.send_text(_RATE_LIMIT_MSG)
        return

    async with deps.locks.get(key):           # serialize one user's messages
        async with deps.semaphore:            # bound global concurrency / LLM spend
            session = Session.load(deps.agent, key, deps.sessions_dir / f"{key}.jsonl")
            if is_flag(msg.text):
                await _handle_flag(msg, key, session, replier, deps)
                return
            if is_help(msg.text):   # greeting / "what can you do" → the grounded capability menu
                await replier.send_text(deps.agent.registry.welcome_text())
                return
            await replier.indicate_typing()
            art_dir = Path(tempfile.mkdtemp(prefix="heynyc-art-"))   # per-request, orchestrator-owned
            try:
                user_drafts = deps.drafts.for_user(key) if deps.drafts else None
                result = await session.send(msg.text, reminders=_reminders(),
                                            output_dir=art_dir, drafts=user_drafts)
                for chunk in render(result):
                    await replier.send_text(chunk)
                for path in _artifacts_in(art_dir):   # only files the tool wrote into OUR dir
                    await replier.send_document(path, caption="Your draft SNAP application (LDSS-4826)")
                analytics.record_interaction(
                    telemetry_path=deps.telemetry_path, model=deps.agent.model,
                    user_key=key, channel=msg.channel, result=result,
                )
            finally:
                shutil.rmtree(art_dir, ignore_errors=True)


async def _handle_flag(msg, key, session, replier, deps) -> None:
    agent_text = _last(session.turns, "assistant")
    if not agent_text:
        await replier.send_text("Nothing to flag yet — ask me something first.")
        return
    analytics.feedback_log(deps.feedback_path, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_key": key,
        "channel": msg.channel,
        "message_id": msg.message_id,
        "flag": msg.text.strip(),
        "user_query": _last(session.turns, "user"),
        "agent_text": agent_text,
    })
    await replier.send_text("Thanks — I've flagged that answer for a human to review. 🙏")
