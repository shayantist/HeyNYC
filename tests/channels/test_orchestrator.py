import asyncio
import pytest
from heynyc.channels.base import InboundMessage, KeyedLocks
from heynyc.channels.orchestrator import (
    Deps,
    handle,
    is_flag,
    is_new,
    is_privacy,
    is_screen,
)
from heynyc.channels.store import ChannelStore
from heynyc.core import config
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry
from heynyc.core.tools import Tool


class FakeReplier:
    def __init__(self):
        self.sent, self.typed = [], 0

    async def send_text(self, text):
        self.sent.append(text)

    async def indicate_typing(self):
        self.typed += 1

    async def send_document(self, path, caption=""):
        return None


def _agent(reply="Here you go."):
    async def complete_fn(messages, tool_schemas):
        return {"role": "assistant", "content": reply, "tool_calls": None}
    return Agent(Registry.discover(config.MODULES_DIR), tools={}, complete_fn=complete_fn, model="fake")


def _deps(tmp_path, **kw):
    store = ChannelStore(tmp_path / "ch.sqlite3", rate_limit=kw.get("rate_limit", 20),
                         window_s=60, dedup_ttl_s=3600)
    return Deps(agent=_agent(kw.get("reply", "Here you go.")), store=store,
                sessions_dir=tmp_path / "sessions", salt="s",
                telemetry_path=tmp_path / "t.jsonl", feedback_path=tmp_path / "fb.jsonl",
                locks=KeyedLocks(), semaphore=asyncio.Semaphore(8))


def _msg(text="when do cooling centers open?", mid="m1"):
    return InboundMessage(channel="whatsapp_meta", sender="+1555", text=text, message_id=mid)


async def test_happy_path_replies_types_and_records(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    await handle(_msg(), replier, deps)
    assert replier.typed == 1
    assert replier.sent and "{cite:" not in replier.sent[0]
    assert (tmp_path / "t.jsonl").exists()


async def test_delivery_failure_does_not_persist_generated_turn(tmp_path):
    class FailingReplier(FakeReplier):
        async def send_text(self, text):
            raise RuntimeError("provider rejected reply")

    deps = _deps(tmp_path)
    with pytest.raises(RuntimeError, match="provider rejected"):
        await handle(_msg(mid="delivery-failure"), FailingReplier(), deps)

    assert not list((tmp_path / "sessions").glob("*.jsonl"))


async def test_document_delivery_failure_does_not_commit_or_record(tmp_path, monkeypatch):
    class FailingReplier(FakeReplier):
        async def send_document(self, path, caption=""):
            raise RuntimeError("provider rejected document")

    artifact = tmp_path / "draft.pdf"
    artifact.write_bytes(b"draft")
    monkeypatch.setattr(
        "heynyc.channels.orchestrator._artifacts_in", lambda _directory: [artifact]
    )

    deps = _deps(tmp_path)
    with pytest.raises(RuntimeError, match="provider rejected document"):
        await handle(_msg(mid="document-failure"), FailingReplier(), deps)

    assert not list((tmp_path / "sessions").glob("*.jsonl"))
    assert not (tmp_path / "t.jsonl").exists()


async def test_compaction_failure_returns_plain_retry_without_persisting(tmp_path):
    async def compact(older, current):
        raise RuntimeError("provider unavailable")

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.agent._memory_limit_tokens = 10
    deps.agent._memory_token_counter = lambda messages, schemas: 11
    deps.agent._memory_compactor = compact

    await handle(_msg(text="continue", mid="compaction-failure"), replier, deps)

    assert "safely fit" in replier.sent[0].lower()
    assert not list((tmp_path / "sessions").glob("*.jsonl"))


async def test_help_intent_returns_capability_menu_without_running_agent(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    await handle(_msg(text="hi"), replier, deps)
    assert replier.typed == 0                         # a greeting short-circuits — no agent run
    assert replier.sent and "How can I help" not in replier.sent[0]
    assert "•" in replier.sent[0]                     # the grounded, example-led capability menu


async def test_media_is_acknowledged_without_claiming_the_model_received_it(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    msg = _msg(text="What does this say?", mid="image-1")
    msg.media = [{"url": "https://api.twilio.com/media/ME1", "content_type": "image/jpeg"}]

    await handle(msg, replier, deps)

    assert replier.typed == 0
    assert replier.sent == [
        "I received the attachment, but this pilot can't read attachments yet. "
        "Please type the text or question you want help with."
    ]


async def test_media_with_imminent_self_harm_text_uses_emergency_backstop_first(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    msg = _msg(text="I'm going to kill myself.", mid="crisis-image-1")
    msg.media = [{"url": "https://api.twilio.com/media/ME1", "content_type": "image/jpeg"}]

    await handle(msg, replier, deps)

    assert replier.typed == 0
    assert replier.sent == [
        "Call 911 right now. Call or text 988 now too. Move away from anything you could use "
        "to hurt yourself and contact someone you trust who can stay with you. I'm an AI and "
        "can't call or monitor emergency help for you."
    ]


async def test_spanish_crisis_short_circuits_before_media_and_model(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()

    async def fail_if_called(messages, tool_schemas):
        raise AssertionError("crisis text must not reach the model")

    deps.agent = Agent(Registry([]), tools={}, complete_fn=fail_if_called, model="fake")
    msg = _msg(text="Quiero matarme.", mid="crisis-spanish-image")
    msg.media = [{"url": "https://api.twilio.com/media/ME1", "content_type": "image/jpeg"}]

    await handle(msg, replier, deps)

    assert replier.typed == 0
    assert "911" in replier.sent[0] and "988" in replier.sent[0]
    assert "adjunto" not in replier.sent[0].lower()


async def test_duplicate_message_is_ignored(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    await handle(_msg(mid="dup"), replier, deps)
    await handle(_msg(mid="dup"), replier, deps)   # same message_id
    assert len(replier.sent) == 1


async def test_rate_limit_blocks_with_a_notice(tmp_path):
    deps = _deps(tmp_path, rate_limit=1)
    r1, r2 = FakeReplier(), FakeReplier()
    await handle(_msg(mid="a"), r1, deps)
    await handle(_msg(mid="b"), r2, deps)
    assert r2.typed == 0 and len(r2.sent) == 1   # a polite "slow down" only


async def test_flag_writes_feedback_and_skips_agent(tmp_path):
    deps = _deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="q1"), FakeReplier(), deps)
    flagger = FakeReplier()
    await handle(_msg(text="wrong", mid="f1"), flagger, deps)
    assert flagger.typed == 0
    assert (tmp_path / "fb.jsonl").exists()
    assert "flagged" in flagger.sent[0].lower()


def test_is_flag():
    assert is_flag("wrong") and is_flag("  Report ") and is_flag("👎")
    assert not is_flag("what's wrong with my application?")


def test_is_screen_only_matches_the_explicit_action_command():
    assert is_screen("/screen")
    assert is_screen("  /SCREEN  ")
    assert is_screen("/screen all")
    assert not is_screen("screen me")
    assert not is_screen("can you screen me for SNAP?")


def test_memory_commands_are_exact_and_do_not_hijack_conversation():
    assert is_new("NEW")
    assert is_new(" /new ")
    assert not is_new("new SNAP rules")
    assert is_privacy("PRIVACY")
    assert is_privacy("/privacy")
    assert not is_privacy("privacy law")


async def test_new_starts_fresh_model_history_without_deleting_audit_file(tmp_path):
    deps = _deps(tmp_path)
    await handle(_msg(text="first question", mid="new-q1"), FakeReplier(), deps)

    replier = FakeReplier()
    await handle(_msg(text="NEW", mid="new-command"), replier, deps)

    session_files = list((tmp_path / "sessions").glob("*.jsonl"))
    assert replier.typed == 0
    assert "new conversation" in replier.sent[0].lower()
    assert len(session_files) == 1
    assert len(session_files[0].read_text().splitlines()) == 3


async def test_privacy_command_is_deterministic_and_does_not_run_agent(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()

    await handle(_msg(text="PRIVACY", mid="privacy-command"), replier, deps)

    text = replier.sent[0].lower()
    assert replier.typed == 0
    assert "encrypted" in text
    assert "30 days" in text
    assert "ai model" in text
    assert "delete my data" not in text
    assert "not yet available" in text
    assert not list((tmp_path / "sessions").glob("*.jsonl"))


async def test_failed_new_acknowledgment_does_not_reset_history(tmp_path):
    class FailingReplier(FakeReplier):
        async def send_text(self, text):
            raise RuntimeError("provider rejected reply")

    deps = _deps(tmp_path)
    await handle(_msg(text="first question", mid="new-before"), FakeReplier(), deps)
    session_file = next((tmp_path / "sessions").glob("*.jsonl"))
    before = session_file.read_text()

    with pytest.raises(RuntimeError, match="provider rejected"):
        await handle(_msg(text="NEW", mid="new-failed"), FailingReplier(), deps)

    assert session_file.read_text() == before


async def test_channel_followup_survives_fresh_dependencies(tmp_path):
    await handle(_msg(text="My first question", mid="restart-before"), FakeReplier(), _deps(tmp_path))

    seen = []

    async def complete_fn(messages, tool_schemas):
        seen.extend(messages)
        return {"role": "assistant", "content": "Follow-up answer", "tool_calls": None}

    deps = _deps(tmp_path)
    deps.agent = Agent(Registry([]), tools={}, complete_fn=complete_fn, model="fake")
    await handle(_msg(text="What about that?", mid="restart-after"), FakeReplier(), deps)

    assert any(message.get("content") == "My first question" for message in seen)
    assert any(message.get("content") == "What about that?" for message in seen)
    assert all(message.get("content") != "Here you go." for message in seen)
    assert any("Prior assistant factual text" in str(message.get("content")) for message in seen)


async def test_screen_command_forces_and_executes_the_screener_through_the_channel(tmp_path):
    calls = []

    async def screen(args, ctx):
        calls.append(args)
        return "official estimate"

    tool = Tool(
        name="screen_eligibility", description="x",
        parameters={"type": "object", "properties": {}}, handler=screen,
    )
    agent = Agent(Registry([]), tools={"screen_eligibility": tool})
    model_calls = []

    async def fake_litellm(messages, tool_schemas, forced_tool=None):
        names = [schema["function"]["name"] for schema in tool_schemas]
        model_calls.append((forced_tool, names))
        message = (
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "function": {
                    "name": "screen_eligibility", "arguments": '{"persons": []}',
                },
            }]}
            if forced_tool
            else {"role": "assistant", "content": "Reply /screen when ready", "tool_calls": None}
        )
        yield {"type": "message", "message": message}

    agent._litellm_stream = fake_litellm
    deps, replier = _deps(tmp_path), FakeReplier()
    deps.agent = agent

    await handle(_msg(text="Here is my complete profile", mid="profile"), replier, deps)
    await handle(_msg(text="/screen", mid="action"), replier, deps)
    await handle(_msg(text="/screen all", mid="action-all"), replier, deps)

    assert model_calls == [
        (None, []),
        ("screen_eligibility", ["screen_eligibility"]),
        (None, ["screen_eligibility"]),
        ("screen_eligibility", ["screen_eligibility"]),
        (None, ["screen_eligibility"]),
    ]
    assert calls == [
        {"persons": [], "show_all": False},
        {"persons": [], "show_all": True},
    ]
    assert replier.sent == [
        "Reply /screen when ready", "Reply /screen when ready", "Reply /screen when ready",
    ]


async def test_per_user_lock_serializes_same_user(tmp_path):
    order = []
    deps = _deps(tmp_path)

    class SlowReplier(FakeReplier):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        async def send_text(self, text):
            order.append(f"{self.tag}-start")
            await asyncio.sleep(0.02)
            order.append(f"{self.tag}-end")
            await super().send_text(text)

    await asyncio.gather(
        handle(_msg(mid="m1"), SlowReplier("A"), deps),
        handle(_msg(mid="m2"), SlowReplier("B"), deps),
    )
    # same user (same sender) → second waits for the first to finish, no interleave
    assert order in (["A-start", "A-end", "B-start", "B-end"],
                     ["B-start", "B-end", "A-start", "A-end"])
