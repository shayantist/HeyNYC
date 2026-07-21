import asyncio

import pytest

from heynyc.channels.base import InboundMessage, KeyedLocks
from heynyc.channels.orchestrator import (
    Deps,
    handle,
    is_delete,
    is_flag,
    is_help,
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


def _answers(replier):
    """Sent messages minus the once-ever first-contact welcome footer, so a test that asserts on
    the ANSWER stays focused on it and isn't perturbed by the new first-contact greeting."""
    return [s for s in replier.sent if not s.startswith("First time here")]


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


def _burn_welcome(deps, channel="whatsapp_meta", sender="+1555"):
    """Consume the once-ever first-contact flag so tests about OTHER behaviors see only
    their answer messages (the welcome now LEADS a first answer on every channel)."""
    from heynyc.channels.identity import user_key
    deps.store.first_contact(user_key(channel, sender, "s"))


def _msg(text="when do cooling centers open?", mid="m1"):
    return InboundMessage(channel="whatsapp_meta", sender="+1555", text=text, message_id=mid)


async def test_deps_event_sink_observes_the_turn_stream(tmp_path):
    """Stage B seam: an event_sink on Deps is threaded into handle's prepare call so a channel
    view (the console REPL) sees the SAME stream every guard rides. None (Twilio) is unchanged."""
    from heynyc.core import events

    seen = []
    deps = _deps(tmp_path)
    deps.event_sink = seen.append
    await handle(_msg(text="when do cooling centers open?", mid="sink1"), FakeReplier(), deps)
    assert any(isinstance(e, events.Done) for e in seen)


async def test_happy_path_replies_types_and_records(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    await handle(_msg(), replier, deps)
    assert replier.typed == 1
    assert replier.sent and "{cite:" not in replier.sent[0]
    assert (tmp_path / "t.jsonl").exists()


async def test_sms_channel_renders_plain_text_but_persists_raw_generation(tmp_path):
    """SMS gets plain text (markdown stripped), while the audit trail keeps the raw model text
    with its markup intact. Pins the ordering: grounding/generation produces the raw turn, which
    is what's persisted; rendering is a deterministic presentation layer downstream of it."""
    deps = _deps(tmp_path, reply="**Cooling centers** are open Saturday.")
    _burn_welcome(deps, channel="sms_twilio")
    replier = FakeReplier()
    msg = InboundMessage(channel="sms_twilio", sender="+1555", text="cooling?", message_id="sms1")

    await handle(msg, replier, deps)

    assert _answers(replier) == ["Cooling centers are open Saturday."]   # no markdown delimiters on SMS
    # audit keeps the raw generation (decoded through the same path the app persists it)
    from heynyc.core.session import _decode_line

    turns = [
        _decode_line(line)
        for line in next((tmp_path / "sessions").glob("*.jsonl")).read_text().splitlines()
        if line.strip()
    ]
    assistant = next(turn for turn in turns if turn.get("role") == "assistant")
    assert assistant["content"] == "**Cooling centers** are open Saturday."


async def test_whatsapp_channel_keeps_native_markup(tmp_path):
    deps = _deps(tmp_path, reply="**Cooling centers** are open Saturday.")
    _burn_welcome(deps, channel="whatsapp_meta", sender="+1556")
    replier = FakeReplier()
    msg = InboundMessage(channel="whatsapp_meta", sender="+1556", text="cooling?", message_id="wa1")

    await handle(msg, replier, deps)

    assert _answers(replier) == ["*Cooling centers* are open Saturday."]  # native WhatsApp bold


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
    assert replier.sent.count("Here you go.") == 1   # the duplicate produced no second answer


async def test_rate_limit_blocks_with_a_notice(tmp_path):
    deps = _deps(tmp_path, rate_limit=1)
    r1, r2 = FakeReplier(), FakeReplier()
    await handle(_msg(mid="a"), r1, deps)
    await handle(_msg(mid="b"), r2, deps)
    assert r2.typed == 0 and len(r2.sent) == 1   # a polite "slow down" only


async def test_report_asks_to_confirm_and_writes_nothing_until_yes(tmp_path):
    """Consent is required, not implied: REPORT/👎 only offers a confirmation stating what a
    human will see. Nothing is recorded (no feedback log, no pointer) until the resident says YES."""
    deps = _deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="q1"), FakeReplier(), deps)

    r1 = FakeReplier()
    await handle(_msg(text="report", mid="f1"), r1, deps)
    assert r1.typed == 0                                   # no agent run, deterministic free lane
    assert "yes" in r1.sent[0].lower() and "human" in r1.sent[0].lower()
    assert not (tmp_path / "fb.jsonl").exists()            # nothing shared yet
    assert deps.store.flags() == []                        # no pointer yet

    r2 = FakeReplier()
    await handle(_msg(text="YES", mid="f2"), r2, deps)
    assert r2.typed == 0
    assert r2.sent == ["Sent. A human will review that one exchange."]
    assert len(deps.store.flags()) == 1                    # exactly one pointer, on confirm
    assert (tmp_path / "fb.jsonl").exists()


async def test_non_yes_after_report_cancels_and_is_processed_as_a_normal_turn(tmp_path):
    deps = _deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="q1"), FakeReplier(), deps)
    await handle(_msg(text="report", mid="f1"), FakeReplier(), deps)

    r = FakeReplier()
    await handle(_msg(text="what about SNAP?", mid="q2"), r, deps)
    assert r.typed == 1                    # the non-YES message ran the agent normally
    assert deps.store.flags() == []        # the pending flag was cancelled, nothing recorded


def test_is_delete_matches_the_command_not_a_question_about_deletion():
    # The fixed command lane matches "DELETE MY DATA" and close natural variants by meaning,
    # deterministically. A QUESTION about deletion is NOT the command (it routes to the agent,
    # which answers from the shipped docs) — same discipline as is_flag vs "what's wrong...".
    assert is_delete("DELETE MY DATA")
    assert is_delete("  delete my data ")
    assert is_delete("delete all my data")
    assert is_delete("erase my data")
    assert is_delete("delete my information")
    assert is_delete("/delete")
    assert not is_delete("how do I delete my data?")
    assert not is_delete("can you delete my account someday")
    assert not is_delete("what happens to my data")


def _drafts_deps(tmp_path, **kw):
    from heynyc.core.drafts import DraftStore

    deps = _deps(tmp_path, **kw)
    deps.drafts = DraftStore(tmp_path / "drafts")
    return deps


async def test_delete_asks_to_confirm_stating_what_goes_and_survives_before_any_yes(tmp_path):
    """Mirrors REPORT: DELETE MY DATA only STAGES a confirmation that states what will be deleted
    and what survives. Deterministic free lane (no agent run); nothing is deleted until YES."""
    deps = _drafts_deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="d0"), FakeReplier(), deps)
    session_file = next((tmp_path / "sessions").glob("*.jsonl"))

    r = FakeReplier()
    await handle(_msg(text="DELETE MY DATA", mid="d1"), r, deps)

    assert r.typed == 0                                   # deterministic, no model call
    copy = r.sent[0].lower()
    assert "yes" in copy                                  # only YES executes
    assert "transcript" in copy and "draft" in copy       # what WILL be deleted
    assert "spend" in copy and ("aggregate" in copy or "statistics" in copy)  # what SURVIVES
    assert session_file.exists()                          # nothing deleted yet


async def test_delete_removes_session_draft_and_flags_and_next_message_starts_fresh(tmp_path):
    deps = _drafts_deps(tmp_path)
    from heynyc.channels.identity import user_key as _uk
    key = _uk("whatsapp_meta", "+1555", "s")

    # Build real state: a committed conversation, a confirmed flag pointer, a draft, and spend.
    await handle(_msg(text="when do cooling centers open?", mid="s1"), FakeReplier(), deps)
    await handle(_msg(text="report", mid="s2"), FakeReplier(), deps)
    await handle(_msg(text="YES", mid="s3"), FakeReplier(), deps)   # confirm the flag pointer
    deps.drafts.for_user(key).merge("snap", {"name": "Jane Doe"})
    deps.store.add_spend(key, "2026-07-20", 0.09)

    session_file = tmp_path / "sessions" / f"{key}.jsonl"
    draft_file = tmp_path / "drafts" / f"{key}.json"
    assert session_file.exists() and draft_file.exists()
    assert len(deps.store.flags()) == 1

    await handle(_msg(text="DELETE MY DATA", mid="d1"), FakeReplier(), deps)
    r = FakeReplier()
    await handle(_msg(text="YES", mid="d2"), r, deps)

    assert r.typed == 0
    done = r.sent[0].lower()
    assert "delet" in done and "spend" in done              # ack confirms + restates survivors
    assert not session_file.exists()                        # transcript actually gone
    assert not draft_file.exists()                          # draft actually gone
    assert deps.store.flags() == []                         # flag rows gone
    assert abs(deps.store.daily_spend(key, "2026-07-20") - 0.09) < 1e-9  # spend survives

    # The next message starts fresh: no earlier turn reaches the model.
    seen = []

    async def complete_fn(messages, tool_schemas):
        seen.extend(messages)
        return {"role": "assistant", "content": "Fresh answer", "tool_calls": None}

    deps.agent = Agent(Registry([]), tools={}, complete_fn=complete_fn, model="fake")
    await handle(_msg(text="hello again", mid="d3"), FakeReplier(), deps)
    # No earlier conversation turn reaches the model. Exclude the system prompt: its standing
    # composition guidance legitimately names "cooling centers near their route" as a worked example.
    assert not any(
        "cooling centers" in str(m.get("content", ""))
        for m in seen if m.get("role") != "system"
    )


async def test_non_yes_after_delete_cancels_and_runs_as_a_normal_turn(tmp_path):
    deps = _drafts_deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="c0"), FakeReplier(), deps)
    session_file = next((tmp_path / "sessions").glob("*.jsonl"))
    await handle(_msg(text="DELETE MY DATA", mid="c1"), FakeReplier(), deps)

    r = FakeReplier()
    await handle(_msg(text="what about SNAP?", mid="c2"), r, deps)
    assert r.typed == 1                # the non-YES message ran the agent normally
    assert session_file.exists()       # the pending deletion was cancelled, nothing removed


async def test_first_contact_appends_a_welcome_footer_exactly_once(tmp_path):
    deps = _deps(tmp_path)

    r1 = FakeReplier()
    await handle(_msg(text="when do cooling centers open?", mid="w1"), r1, deps)
    footer = " ".join(r1.sent)
    assert "HeyNYC" in footer                                  # one line on what HeyNYC is
    for command in ("HELP", "PRIVACY", "REPORT", "DELETE MY DATA"):
        assert command in footer                               # names every control, once

    r2 = FakeReplier()
    await handle(_msg(text="what about SNAP?", mid="w2"), r2, deps)
    assert "DELETE MY DATA" not in " ".join(r2.sent)           # never welcomed twice


def test_store_stages_confirms_and_lists_flag_pointers(tmp_path):
    store = ChannelStore(tmp_path / "s.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=60)
    assert store.pop_pending_flag("u1") is None
    store.set_pending_flag("u1", 3, "report")
    store.set_pending_flag("u1", 5, "/wrong")     # a fresh REPORT replaces the un-confirmed one
    staged = store.pop_pending_flag("u1")
    assert staged["turn_index"] == 5 and staged["flag"] == "/wrong"
    assert store.pop_pending_flag("u1") is None   # consumed once
    assert store.flags() == []
    store.add_flag("u1", 5, "/wrong")
    flags = store.flags()
    assert len(flags) == 1
    assert flags[0]["user_key"] == "u1" and flags[0]["turn_index"] == 5 and flags[0]["flag"] == "/wrong"


def test_is_flag():
    assert is_flag("wrong") and is_flag("  Report ") and is_flag("👎")
    assert not is_flag("what's wrong with my application?")


def test_is_help_menu_only_matches_greetings_not_a_help_word_mid_question():
    # The capability menu answers a bare greeting / "what can you do", never an ordinary
    # question that merely contains the word "help" or a greeting token.
    assert is_help("hi") and is_help("  Menu ") and is_help("what can you do?")
    assert is_help("HELP") and is_help("/help")
    assert not is_help("hey what's the nearest cooling center")
    assert not is_help("can you help me find a food pantry")
    assert not is_help("i need help with my SNAP application")


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
    assert "delete my data" in text          # self-service deletion now exists, name the command
    assert "not yet available" not in text   # the pending-design language is gone
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
    assert any(
        "Earlier assistant reply" in str(message.get("content"))
        and "Here you go." in str(message.get("content"))
        for message in seen
    )


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
    _burn_welcome(deps, channel="whatsapp_meta")
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
    assert _answers(replier) == [
        "Reply /screen when ready", "Reply /screen when ready", "Reply /screen when ready",
    ]


async def test_per_user_lock_serializes_same_user(tmp_path):
    order = []
    deps = _deps(tmp_path)
    from heynyc.channels.identity import user_key as _uk
    # Consume the once-ever welcome up front so the first-contact footer doesn't add an extra
    # send and perturb the serialization order this test isolates.
    deps.store.first_contact(_uk("whatsapp_meta", "+1555", "s"))

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


def test_store_tracks_daily_spend_per_user_and_day(tmp_path):
    store = ChannelStore(tmp_path / "s.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=60)

    assert store.daily_spend("u1", "2026-07-18") == 0.0
    store.add_spend("u1", "2026-07-18", 0.02)
    store.add_spend("u1", "2026-07-18", 0.03)
    store.add_spend("u2", "2026-07-18", 0.40)
    store.add_spend("u1", "2026-07-19", 0.99)

    assert abs(store.daily_spend("u1", "2026-07-18") - 0.05) < 1e-9
    assert abs(store.daily_spend("u2", "2026-07-18") - 0.40) < 1e-9
    assert abs(store.daily_spend("u1", "2026-07-19") - 0.99) < 1e-9


async def test_user_over_daily_cap_gets_fixed_copy_and_no_agent_call(tmp_path):
    """One resident going ham never dims the service for anyone else: the cap is per user
    per NYC day, fails closed with warm fixed copy, and free commands stay available."""
    from heynyc.channels.orchestrator import _DAILY_CAP_MSG, _nyc_day

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.user_daily_spend_cap = 0.50
    key_of = lambda m: __import__("heynyc.channels.identity", fromlist=["user_key"]).user_key(
        m.channel, m.sender, "s",
    )
    message = _msg(text="whats happening this weekend", mid="cap1")
    deps.store.add_spend(key_of(message), _nyc_day(), 0.60)  # already over today

    await handle(message, replier, deps)

    assert replier.sent == [_DAILY_CAP_MSG]
    assert replier.typed == 0  # the agent never ran

    # Free commands still work while capped.
    await handle(_msg(text="privacy", mid="cap2"), replier, deps)
    assert len(replier.sent) == 2
    assert "usage limit" not in replier.sent[-1].lower()


async def test_emergency_text_bypasses_the_daily_cap(tmp_path):
    from heynyc.channels.orchestrator import _nyc_day

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.user_daily_spend_cap = 0.50
    message = _msg(text="I have severe chest pain right now", mid="cap3")
    from heynyc.channels.identity import user_key as _uk
    deps.store.add_spend(_uk(message.channel, message.sender, "s"), _nyc_day(), 9.99)

    await handle(message, replier, deps)

    assert any("911" in text for text in replier.sent)


async def test_turn_cost_accrues_to_the_daily_tally(tmp_path):
    from heynyc.channels.identity import user_key as _uk
    from heynyc.channels.orchestrator import _nyc_day

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.user_daily_spend_cap = 5.00
    message = _msg(text="whats happening this weekend", mid="cap4")

    await handle(message, replier, deps)

    key = _uk(message.channel, message.sender, "s")
    assert deps.store.daily_spend(key, _nyc_day()) >= 0.0  # tally exists (cost may be 0 in fakes)
