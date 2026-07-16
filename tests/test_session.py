from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from heynyc.core import pii_crypto
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry
from heynyc.core.session import Session, migrate_plaintext_sessions, purge_expired_sessions


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


# --- Encryption at rest (security-audit F1) ---------------------------------


async def test_transcript_round_trips_and_hides_pii_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    path = tmp_path / "enc.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("noted"))
    s = Session(agent=agent, id="enc", path=path)
    await s.send("my address is 123 Main Street")

    raw = path.read_bytes()
    assert b"123 Main Street" not in raw  # the typed PII is not on disk in the clear

    # a fresh load with the key transparently decrypts the multi-turn context
    resumed = Session.load(agent, "enc", path)
    assert [t["content"] for t in resumed.turns] == ["my address is 123 Main Street", "noted"]


async def test_multi_turn_context_survives_encryption(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    path = tmp_path / "multi.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="multi", path=path)
    await s.send("turn one")
    await s.send("turn two")
    resumed = Session.load(agent, "multi", path)
    assert [t["content"] for t in resumed.turns] == ["turn one", "ok", "turn two", "ok"]


async def test_transcript_stays_cleartext_when_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)  # the dev path
    path = tmp_path / "plain.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="plain", path=path)
    await s.send("hello")
    assert "hello" in path.read_text()  # unchanged dev behavior


async def test_plaintext_session_is_migrated_when_encryption_is_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)
    path = tmp_path / "legacy.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("legacy answer"))
    await Session(agent=agent, id="legacy", path=path).send("Ana Diaz needs SNAP")
    assert "Ana Diaz" in path.read_text()

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    assert migrate_plaintext_sessions(tmp_path) == [str(path)]
    assert b"Ana Diaz" not in path.read_bytes()
    resumed = Session.load(agent, "legacy", path)
    assert [turn["content"] for turn in resumed.turns] == [
        "Ana Diaz needs SNAP", "legacy answer",
    ]


# --- Retention / TTL sweep (irreversible; GDPR Art 5(1)(e)) -----------------


def _age_file(path, days: float) -> None:
    old = time.time() - days * 86400
    os.utime(path, (old, old))


async def test_purge_expired_sessions_deletes_old_keeps_recent(tmp_path):
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    for name in ("old", "new"):
        s = Session(agent=agent, id=name, path=tmp_path / f"{name}.jsonl")
        await s.send("hi")
    _age_file(tmp_path / "old.jsonl", days=45)
    deleted = purge_expired_sessions(tmp_path, max_age_days=30)
    assert not (tmp_path / "old.jsonl").exists()  # irreversibly gone
    assert (tmp_path / "new.jsonl").exists()
    assert any("old.jsonl" in p for p in deleted)
