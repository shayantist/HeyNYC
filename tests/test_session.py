from __future__ import annotations

from pathlib import Path

import pytest

from heynyc.core.agent import Agent
from heynyc.core.registry import Registry
from heynyc.core.session import Session


def _const_complete(text: str):
    async def fn(messages, schemas):
        return {"role": "assistant", "content": text, "tool_calls": None}

    return fn


async def test_session_persists_and_resumes(tmp_path: Path):
    path = tmp_path / "s1.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("first answer"))

    s1 = Session(agent=agent, id="s1", path=path)
    await s1.send("first question")
    assert path.exists()
    assert len(s1.turns) == 2

    # New session object, same file → history restored
    s2 = Session.load(agent, "s1", path)
    assert [t["content"] for t in s2.turns] == ["first question", "first answer"]


async def test_session_appends_across_turns(tmp_path: Path):
    path = tmp_path / "s2.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="s2", path=path)
    await s.send("q1")
    await s.send("q2")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # 2 turns × (user + assistant)


async def test_session_stream_persists(tmp_path: Path):
    path = tmp_path / "s3.jsonl"

    async def sf(messages, schemas):
        yield {"type": "text", "text": "streamed"}
        yield {"type": "message", "message": {"role": "assistant", "content": "streamed", "tool_calls": None}}

    agent = Agent(Registry([]), tools={}, stream_fn=sf)
    s = Session(agent=agent, id="s3", path=path)
    types = [e.type async for e in s.stream("hi")]
    assert types[-1] == "done"
    assert len(s.turns) == 2
    assert path.exists()


async def test_session_no_path_is_memory_only():
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="mem")
    await s.send("q")
    assert len(s.turns) == 2  # works, just not persisted
