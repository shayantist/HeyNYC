from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from heynyc.core import pii_crypto
from heynyc.core.agent import Agent
from heynyc.core.memory import ContinuityRecord
from heynyc.core.registry import Registry
from heynyc.core.session import (
    Session,
    migrate_plaintext_sessions,
    purge_expired_sessions,
)
from heynyc.core.tools import Tool


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


async def test_session_persists_grounded_assistant_evidence(tmp_path: Path):
    path = tmp_path / "grounded.jsonl"
    responses = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "function": {"name": "lookup", "arguments": "{}"},
        }]},
        {"role": "assistant", "content": "Verified result {cite:S1}", "tool_calls": None},
    ]

    async def complete(messages, schemas):
        return responses.pop(0)

    async def lookup(args, ctx):
        cid = ctx.citations.register(
            "https://data.cityofnewyork.us/example",
            snippet="Verified result",
            title="Verified example",
            kind="DATA",
        )
        return f"Verified result {{cite:{cid}}}"

    agent = Agent(
        Registry([]),
        tools={"lookup": Tool("lookup", "", {}, lookup)},
        complete_fn=complete,
    )
    await Session(agent=agent, id="grounded", path=path).send("look it up")

    resumed = Session.load(agent, "grounded", path)
    assert resumed.turns[-1]["citations"]["S1"]["title"] == "Verified example"


async def test_session_appends_across_turns(tmp_path: Path):
    path = tmp_path / "s2.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="s2", path=path)
    await s.send("q1")
    await s.send("q2")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # 2 turns × (user + assistant)


async def test_session_no_path_is_memory_only():
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    s = Session(agent=agent, id="mem")
    await s.send("q")
    assert len(s.turns) == 2  # works, just not persisted


async def test_prepared_turn_is_not_visible_or_persisted_until_committed(tmp_path: Path):
    path = tmp_path / "delivery.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("not delivered yet"))
    session = Session(agent=agent, id="delivery", path=path)

    pending = await session.prepare("question")

    assert session.turns == []
    assert not path.exists()

    session.commit(pending)

    assert [turn["content"] for turn in session.turns] == ["question", "not delivered yet"]
    assert path.exists()


async def test_committed_turns_are_stamped_and_stamps_survive_reload(tmp_path: Path):
    """F062: the sent time persists with each turn so a resumed session (hours or days later)
    labels prior replies with when they were actually sent."""
    from datetime import datetime

    path = tmp_path / "stamped.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("ok"))
    session = Session(agent=agent, id="stamped", path=path)
    await session.send("q1")

    resumed = Session.load(agent, "stamped", path)
    assert len(resumed.turns) == 2
    for turn in resumed.turns:
        sent = datetime.fromisoformat(turn["timestamp"])
        assert sent.utcoffset() is not None


async def test_new_reset_boundary_preserves_audit_file_but_clears_model_history(tmp_path: Path):
    path = tmp_path / "reset.jsonl"
    agent = Agent(Registry([]), tools={}, complete_fn=_const_complete("answer"))
    session = Session(agent=agent, id="reset", path=path)
    await session.send("old question")

    session.reset()

    assert session.turns == []
    assert path.exists()
    resumed = Session.load(agent, "reset", path)
    assert resumed.turns == []
    assert len(path.read_text().splitlines()) == 3


async def test_session_compacts_only_under_pressure_and_persists_typed_continuity(tmp_path: Path):
    path = tmp_path / "compact.jsonl"
    seen_messages = []
    compact_calls = []

    async def complete(messages, schemas):
        seen_messages.append(messages)
        return {"role": "assistant", "content": "continued", "tool_calls": None}

    async def compact(older, current):
        compact_calls.append(older)
        return ContinuityRecord(
            goal="I need food help",
            exact_user_excerpts=["I need food help"],
        )

    def count(messages, schemas):
        return sum(
            len(str(message.get("content") or ""))
            for message in messages
            if message.get("role") != "system"
            and not str(message.get("content") or "").startswith("<system-reminder>")
        )

    agent = Agent(
        Registry([]), tools={}, complete_fn=complete,
        memory_limit_tokens=300, memory_token_counter=count, memory_compactor=compact,
    )
    session = Session(agent=agent, id="compact", path=path)
    session.convo.turns = [
        {"role": "user", "content": "I need food help " * 30},
        {"role": "assistant", "content": "Tell me what changed " * 20},
        {"role": "user", "content": "Queens"},
        {"role": "assistant", "content": "I can help with that"},
    ]

    pending = await session.prepare("What is the next step?")
    session.commit(pending)

    assert len(compact_calls) == 1
    assert any("I need food help" in str(message.get("content")) for message in seen_messages[-1])
    assert all("Tell me what changed" not in str(message.get("content")) for message in seen_messages[-1])
    resumed = Session.load(agent, "compact", path)
    assert resumed.continuity.goal == "I need food help"
    assert pending.result.usage["memory_compactions"] == 1


async def test_late_awareness_is_measured_before_memory_planning(tmp_path: Path):
    compact_calls = []

    async def awareness():
        return "citywide alert"

    async def compact(older, current):
        compact_calls.append(older)
        return ContinuityRecord(goal="first question")

    def count(messages, schemas):
        text = " ".join(str(message.get("content") or "") for message in messages)
        if "citywide alert" in text and text.count("Earlier assistant reply") >= 2:
            return 11
        return 9

    agent = Agent(
        Registry([]), tools={}, complete_fn=_const_complete("continued"),
        notify_awareness=awareness, memory_limit_tokens=10,
        memory_token_counter=count, memory_compactor=compact,
    )
    session = Session(agent=agent, id="awareness", path=tmp_path / "awareness.jsonl")
    session.convo.turns = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    pending = await session.prepare("continue")

    assert pending.result.status == "success"
    assert compact_calls


async def test_context_limit_is_not_returned_as_a_committable_turn(tmp_path: Path):
    agent = Agent(
        Registry([]), tools={}, complete_fn=_const_complete("should not run"),
        memory_limit_tokens=1, memory_token_counter=lambda messages, schemas: 2,
    )
    session = Session(agent=agent, id="limit", path=tmp_path / "limit.jsonl")

    with pytest.raises(Exception, match="context"):
        await session.prepare("hello")

    assert session.turns == []
    assert not (tmp_path / "limit.jsonl").exists()


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
