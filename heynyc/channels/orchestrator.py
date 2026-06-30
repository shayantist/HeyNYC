"""The universal, channel-agnostic flow: dedup -> rate-limit -> per-user lock ->
bounded concurrency -> run the agent on the user's session -> reply -> record.
Flag keywords short-circuit into the feedback log instead of the agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


def is_flag(text: str) -> bool:
    return text.strip().lower() in _FLAG_TOKENS


def _reminders() -> list[str]:
    return [f"Today's date is {datetime.now():%A, %B %-d, %Y}. The user is in New York City."]


def _last(turns: list[dict], role: str) -> str:
    for turn in reversed(turns):
        if turn.get("role") == role:
            return turn.get("content") or ""
    return ""


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
            await replier.indicate_typing()
            result = await session.send(msg.text, reminders=_reminders())
            for chunk in render(result):
                await replier.send_text(chunk)
            analytics.record_interaction(
                telemetry_path=deps.telemetry_path, model=deps.agent.model,
                user_key=key, channel=msg.channel, result=result,
            )


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
