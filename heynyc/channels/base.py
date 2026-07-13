"""The channel-agnostic port: a normalized inbound message, a provider-specific
replier, and the fire-and-forget dispatch seam (swap for a Celery/Dramatiq queue later)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger("heynyc.channels")

_TASKS: set[asyncio.Task] = set()


@dataclass
class InboundMessage:
    channel: str          # "whatsapp_meta" | "whatsapp_twilio" | "sms_twilio"
    sender: str           # raw address, in-memory only
    text: str
    message_id: str       # wamid / MessageSid, the dedup key
    profile_name: str = ""
    media: list[dict] = field(default_factory=list)   # v2 voice notes ride here
    raw: dict = field(default_factory=dict)


@runtime_checkable
class Replier(Protocol):
    async def send_text(self, text: str) -> None: ...
    async def indicate_typing(self) -> None: ...      # no-op on channels without it
    async def send_document(self, path: str, caption: str = "") -> None: ...  # e.g. a filled PDF


async def _guard(coro) -> None:
    try:
        await coro
    except Exception:
        logger.exception("dispatched channel task failed")


def dispatch(coro) -> None:
    """Run `coro` in the background, return immediately. Retains a reference so the
    task isn't GC'd, and logs (never raises) on failure, nothing awaits the result."""
    task = asyncio.create_task(_guard(coro))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def drain() -> None:
    """Await all outstanding dispatched tasks (graceful shutdown)."""
    if _TASKS:
        await asyncio.gather(*list(_TASKS), return_exceptions=True)


class KeyedLocks:
    """One asyncio.Lock per key, created lazily, serializes a single user's messages."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock
