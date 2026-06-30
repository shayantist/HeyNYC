import asyncio
import pytest
from heynyc.channels.base import InboundMessage, dispatch, drain, KeyedLocks


async def test_dispatch_runs_and_drain_awaits():
    seen = []

    async def work():
        await asyncio.sleep(0.01)
        seen.append("done")

    dispatch(work())
    assert seen == []      # fire-and-forget: hasn't completed synchronously
    await drain()
    assert seen == ["done"]


async def test_dispatch_swallows_and_logs_exceptions():
    async def boom():
        raise ValueError("nope")

    dispatch(boom())       # must not raise
    await drain()          # must not raise


def test_keyed_locks_are_per_key():
    locks = KeyedLocks()
    assert locks.get("a") is locks.get("a")
    assert locks.get("a") is not locks.get("b")


def test_inbound_message_defaults():
    m = InboundMessage(channel="c", sender="s", text="hi", message_id="m1")
    assert m.profile_name == "" and m.media == [] and m.raw == {}
