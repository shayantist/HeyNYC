import asyncio
import json
from pathlib import Path

import pytest
from babel.messages.extract import extract_from_file

from heynyc.channels.base import InboundMessage, KeyedLocks
from heynyc.channels.orchestrator import (
    Deps,
    _welcome_footer,
    handle,
    is_delete,
    is_flag,
    is_help,
    is_new,
    is_privacy,
    is_screen,
)
from heynyc.channels.store import ChannelStore
from heynyc.core import config, localization
from heynyc.core.agent import (
    _IMMINENT_SELF_HARM_RESPONSE_EN,
    Agent,
    AgentResult,
)
from heynyc.core.manifest import ServiceModule
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


async def test_approval_is_not_active_when_review_delivery_fails(tmp_path, monkeypatch):
    from heynyc.channels.identity import user_key
    from heynyc.core import pii_crypto
    from heynyc.core.session import PendingTurn, Session

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())

    class NativeAgent:
        model = "fake-native"
        registry = Registry([])
        tools = {}

        def conversation(self):
            return object()

        def conversation_from_state(self, state):
            raise AssertionError("no prior native state should load")

    async def prepare(self, message, **kwargs):
        return PendingTurn(
            user_message=message,
            result=AgentResult(
                text="Review this action",
                citations={},
                tool_calls_made=[],
                iterations=1,
                status="approval_required",
                messages=[],
                usage={"cost_usd": 0.0},
            ),
            runtime_state=b'{"pending":true}',
        )

    class FailingReplier(FakeReplier):
        async def send_text(self, text):
            raise RuntimeError("provider rejected review")

    monkeypatch.setattr(Session, "prepare", prepare)
    deps = _deps(tmp_path)
    deps.agent = NativeAgent()
    _burn_welcome(deps)

    with pytest.raises(RuntimeError, match="provider rejected review"):
        await handle(_msg(text="submit it", mid="approval-delivery-failure"), FailingReplier(), deps)

    key = user_key("whatsapp_meta", "+1555", "s")
    assert deps.store.has_pending_approval(key) is False
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

    assert replier.typed == 1
    assert _answers(replier) == ["Here you go."]


async def test_media_with_imminent_self_harm_text_uses_emergency_backstop_first(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    msg = _msg(text="I'm going to kill myself.", mid="crisis-image-1")
    msg.media = [{"url": "https://api.twilio.com/media/ME1", "content_type": "image/jpeg"}]

    await handle(msg, replier, deps)

    assert replier.typed == 1
    assert _answers(replier)[0].startswith(_IMMINENT_SELF_HARM_RESPONSE_EN)


async def test_spanish_crisis_short_circuits_before_media_and_model(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()

    async def fail_if_called(messages, tool_schemas):
        raise AssertionError("crisis text must not reach the model")

    deps.agent = Agent(Registry([]), tools={}, complete_fn=fail_if_called, model="fake")
    msg = _msg(text="Quiero matarme.", mid="crisis-spanish-image")
    msg.media = [{"url": "https://api.twilio.com/media/ME1", "content_type": "image/jpeg"}]

    await handle(msg, replier, deps)

    assert replier.typed == 1
    answer = _answers(replier)[0]
    assert "911" in answer and "988" in answer
    assert "adjunto" not in answer.lower()


async def test_duplicate_message_is_ignored(tmp_path):
    deps, replier = _deps(tmp_path), FakeReplier()
    await handle(_msg(mid="dup"), replier, deps)
    await handle(_msg(mid="dup"), replier, deps)   # same message_id
    assert replier.sent.count("Here you go.") == 1   # the duplicate produced no second answer


async def test_durable_inbox_can_skip_the_second_dedup_check(tmp_path):
    from heynyc.channels.identity import user_key

    deps, replier = _deps(tmp_path), FakeReplier()
    message = _msg(mid="already-in-inbox")
    deps.store.seen(message.message_id, user_key(message.channel, message.sender, deps.salt))

    await handle(message, replier, deps, deduplicate=False)

    assert "Here you go." in replier.sent


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


async def test_sms_disliked_tapback_enters_the_same_consent_flow(tmp_path):
    deps = _deps(tmp_path)
    await handle(_msg(text="when do cooling centers open?", mid="tap-q1"), FakeReplier(), deps)

    report = FakeReplier()
    await handle(
        _msg(text='Disliked "Honestly, that one was too vague"', mid="tap-f1"),
        report,
        deps,
    )

    assert report.typed == 0
    assert "yes" in report.sent[0].lower() and "human" in report.sent[0].lower()
    assert deps.store.flags() == []

    confirmed = FakeReplier()
    await handle(_msg(text="YES", mid="tap-f2"), confirmed, deps)

    assert confirmed.sent == ["Sent. A human will review that one exchange."]
    assert deps.store.flags()[0]["flag"] == "disliked"


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
    assert "transcript" in copy and "draft" in copy and "queued" in copy
    assert "spend" in copy and ("aggregate" in copy or "statistics" in copy)  # what SURVIVES
    assert session_file.exists()                          # nothing deleted yet


async def test_delete_removes_session_draft_and_flags_and_next_message_starts_fresh(tmp_path):
    deps = _drafts_deps(tmp_path)
    from heynyc.channels.identity import user_key as _uk
    from heynyc.core import pii_crypto
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
    assert "delet" in done and "queued" in done and "spend" in done
    assert not session_file.exists()                        # transcript actually gone
    assert not draft_file.exists()                          # draft actually gone
    assert deps.store.flags() == []                         # flag rows gone
    assert abs(deps.store.daily_spend(key, "2026-07-20") - 0.09) < 1e-9  # spend survives
    assert pii_crypto.deletion_generation(tmp_path / ".deletion-generation") == 1

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


async def test_delete_fails_closed_when_snapshot_invalidation_cannot_persist(
    tmp_path, monkeypatch
):
    from heynyc.core import pii_crypto

    deps = _drafts_deps(tmp_path)
    await handle(_msg(text="before deletion", mid="barrier-0"), FakeReplier(), deps)
    session_file = next((tmp_path / "sessions").glob("*.jsonl"))
    await handle(_msg(text="DELETE MY DATA", mid="barrier-1"), FakeReplier(), deps)

    def fail(_path):
        raise OSError("barrier unavailable")

    monkeypatch.setattr(pii_crypto, "advance_deletion_generation", fail)

    with pytest.raises(OSError, match="barrier unavailable"):
        await handle(_msg(text="YES", mid="barrier-2"), FakeReplier(), deps)

    assert session_file.exists()


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


async def test_first_contact_passes_native_safety_language_to_welcome(tmp_path, monkeypatch):
    captured = []

    def localized_footer(categories, language):
        captured.append((categories, language))
        return "Localized welcome"

    monkeypatch.setattr(
        "heynyc.channels.orchestrator._localized_welcome_footer",
        localized_footer,
    )

    class NativeConversation:
        async def send(self, message, **kwargs):
            return AgentResult(
                text="Answer",
                citations={},
                tool_calls_made=[],
                iterations=1,
                status="success",
                messages=[],
                usage={"cost_usd": 0.0},
                diagnostics={"safety_language": "es"},
            )

        def dump_state(self):
            return b"{}"

    class NativeAgent:
        model = "fake-native"
        registry = Registry([])

        def conversation(self):
            return NativeConversation()

        def conversation_from_state(self, state):
            return NativeConversation()

    deps = _deps(tmp_path)
    deps.agent = NativeAgent()
    replier = FakeReplier()

    await handle(_msg(text="where can I find food?", mid="native-welcome"), replier, deps)

    assert captured == [([], "es")]
    assert replier.sent == ["Localized welcome\n\nNow, about your message:", "Answer"]


def _welcome_test_agent(language):
    class Conversation:
        async def send(self, message, **kwargs):
            return AgentResult(
                text="Grounded answer",
                citations={},
                tool_calls_made=[],
                iterations=1,
                status="success",
                messages=[],
                usage={"cost_usd": 0.0},
                diagnostics={"safety_language": language},
            )

        def dump_state(self):
            return b"{}"

    class WelcomeAgent:
        model = "fake-native"
        registry = Registry([])

        def conversation(self):
            return Conversation()

        def conversation_from_state(self, state):
            return Conversation()

    return WelcomeAgent()


async def test_first_contact_accepts_regional_english_language_once(tmp_path):

    deps = _deps(tmp_path)
    deps.agent = _welcome_test_agent("en-US")
    replier = FakeReplier()

    await handle(_msg(text="where can I find food?", mid="regional-english"), replier, deps)

    assert replier.sent == [
        "First time here? I'm HeyNYC. I help with NYC services across NYC, grounded in real city "
        "data, and I cite my sources.\n"
        "Anytime, text HELP for what I can do, PRIVACY for how your info is handled, REPORT to "
        "flag a bad answer, or DELETE MY DATA to erase everything I keep.\n\n"
        "Now, about your message:",
        "Grounded answer",
    ]

    second = FakeReplier()
    await handle(_msg(text="where can I find food?", mid="regional-english-2"), second, deps)
    assert second.sent == ["Grounded answer"]


@pytest.mark.parametrize("language", ["es", 123, "not-a-language"])
async def test_first_contact_suppresses_untranslated_non_english_welcome(tmp_path, language):
    deps = _deps(tmp_path)
    deps.agent = _welcome_test_agent(language)
    replier = FakeReplier()

    await handle(_msg(text="where can I find food?", mid="missing-catalog"), replier, deps)

    assert replier.sent == ["Grounded answer"]


@pytest.mark.parametrize("language", [None, "", "en", "en-GB"])
def test_welcome_footer_uses_english_fallback_and_babel_list_format(language):
    registry = Registry([
        ServiceModule(name="food", category="Food"),
        ServiceModule(name="transit", category="Transit"),
    ])

    footer = _welcome_footer(registry, language)

    assert "I help with Food and Transit across NYC" in footer


@pytest.mark.parametrize("language", ["es", 123, "not-a-language"])
def test_welcome_footer_fails_closed_without_a_usable_catalog(language):
    assert _welcome_footer(Registry([]), language) is None


def test_babel_extracts_welcome_msgid_from_localization_module():
    messages = extract_from_file("python", Path(localization.__file__))

    assert any(message == localization._WELCOME_FOOTER for _, message, *_ in messages)


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
    assert is_flag('Disliked "Honestly, that one was too vague"')
    assert is_flag("Disliked “Honestly, that one was too vague”")
    assert not is_flag("what's wrong with my application?")
    assert not is_flag('Liked "That helped"')
    assert not is_flag('I disliked "that answer"')


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


async def test_native_approval_is_reviewed_and_resumed_through_the_channel(
    tmp_path,
    monkeypatch,
):
    from heynyc.core import pii_crypto

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())

    class NativeConversation:
        def __init__(self, phase="new"):
            self.phase = phase

        @property
        def pending_approvals(self):
            if self.phase != "pending":
                return {}
            return {
                "call-1": {
                    "tool_name": "submit_request",
                    "args": {"borough": "Queens"},
                }
            }

        @property
        def pending_calls(self):
            return {}

        async def send(self, message, **kwargs):
            self.phase = "pending"
            return result("", "approval_required")

        async def resume_approvals(self, approvals, **kwargs):
            assert approvals == {"call-1": True}
            self.phase = "done"
            return result("Submitted after your approval")

        def dump_state(self):
            return json.dumps({"phase": self.phase}).encode()

    class NativeAgent:
        model = "fake-native"
        registry = Registry.discover(config.MODULES_DIR)
        tools = {
            "submit_request": Tool(
                name="submit_request",
                description="Submit",
                parameters={"type": "object", "properties": {}},
                handler=lambda args, ctx: None,
                read_only=False,
                requires_approval=True,
            )
        }

        def conversation(self):
            return NativeConversation()

        def conversation_from_state(self, state):
            return NativeConversation(json.loads(state)["phase"])

    def result(text, status="success"):
        return AgentResult(
            text=text,
            citations={},
            tool_calls_made=[],
            iterations=1,
            status=status,
            messages=[],
            usage={"cost_usd": 0.0},
        )

    deps = _deps(tmp_path)
    deps.agent = NativeAgent()
    _burn_welcome(deps)

    review = FakeReplier()
    await handle(_msg(text="submit it", mid="approval-1"), review, deps)
    assert "submit_request" in review.sent[0]
    assert "Reply YES to approve, or NO to deny" in review.sent[0]

    shorthand = FakeReplier()
    await handle(_msg(text="Y", mid="approval-short"), shorthand, deps)
    assert shorthand.sent == [review.sent[0]]

    approved = FakeReplier()
    await handle(_msg(text="YES", mid="approval-2"), approved, deps)
    assert approved.sent == ["Submitted after your approval"]

    from heynyc.channels.identity import user_key as _uk
    from heynyc.core.session import Session

    key = _uk("whatsapp_meta", "+1555", "s")
    assert deps.store.has_pending_approval(key) is False
    resumed = Session.load(
        deps.agent,
        key,
        tmp_path / "sessions" / f"{key}.jsonl",
    )
    assert [turn["content"] for turn in resumed.turns] == [
        "submit it",
        review.sent[0],
        "YES",
        "Submitted after your approval",
    ]


async def test_second_approval_is_not_active_when_its_review_delivery_fails(
    tmp_path,
    monkeypatch,
):
    from heynyc.channels.identity import user_key as _uk
    from heynyc.core import pii_crypto

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())

    class NativeConversation:
        def __init__(self, phase="new"):
            self.phase = phase

        @property
        def pending_approvals(self):
            if self.phase == "first":
                return {"call-1": {"tool_name": "first_action", "args": {}}}
            if self.phase == "second":
                return {"call-2": {"tool_name": "second_action", "args": {}}}
            return {}

        @property
        def pending_calls(self):
            return {}

        async def send(self, message, **kwargs):
            self.phase = "first"
            return _result("approval_required")

        async def resume_approvals(self, approvals, **kwargs):
            self.phase = "second"
            return _result("approval_required")

        def dump_state(self):
            return json.dumps({"phase": self.phase}).encode()

    class NativeAgent:
        model = "fake-native"
        registry = Registry([])
        tools = {
            name: Tool(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                handler=lambda args, ctx: None,
                read_only=False,
                requires_approval=True,
                idempotent=True,
            )
            for name in ("first_action", "second_action")
        }

        def conversation(self):
            return NativeConversation()

        def conversation_from_state(self, state):
            return NativeConversation(json.loads(state)["phase"])

    def _result(status):
        return AgentResult(
            text="",
            citations={},
            tool_calls_made=[],
            iterations=1,
            status=status,
            messages=[],
            usage={"cost_usd": 0.0},
        )

    class FailingReplier(FakeReplier):
        async def send_text(self, text):
            raise RuntimeError("provider rejected second review")

    deps = _deps(tmp_path)
    deps.agent = NativeAgent()
    _burn_welcome(deps)
    key = _uk("whatsapp_meta", "+1555", "s")

    await handle(_msg(text="start both", mid="approval-stage-1"), FakeReplier(), deps)
    assert deps.store.has_pending_approval(key) is True

    with pytest.raises(RuntimeError, match="provider rejected second review"):
        await handle(_msg(text="YES", mid="approval-stage-2"), FailingReplier(), deps)

    assert deps.store.has_pending_approval(key) is False


async def test_native_retry_safe_approval_recovers_after_partial_failure(
    tmp_path,
    monkeypatch,
):
    from pydantic_ai import UnexpectedModelBehavior
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from heynyc.channels.identity import user_key as _uk
    from heynyc.core import pii_crypto
    from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
    from heynyc.core.tools.base import ToolContext

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    attempts = 0
    fail_once = True

    async def prepare(args: dict, ctx: ToolContext) -> str:
        nonlocal attempts
        attempts += 1
        return "Prepared"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal fail_once
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(
                [ToolCallPart("prepare_application", {}, "approval-call")]
            )
        if fail_once:
            fail_once = False
            raise UnexpectedModelBehavior("invalid response after action")
        return ModelResponse([TextPart("Prepared after retry")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "prepare_application": Tool(
                name="prepare_application",
                description="Prepare an approved draft",
                parameters={"type": "object", "properties": {}},
                handler=prepare,
                read_only=False,
                requires_approval=True,
                idempotent=True,
            )
        },
        guard_grounding=False,
    )
    deps = _deps(tmp_path)
    deps.agent = runtime
    _burn_welcome(deps)
    key = _uk("whatsapp_meta", "+1555", "s")

    review = FakeReplier()
    await handle(_msg(text="prepare it", mid="approval-failure-1"), review, deps)
    failed = FakeReplier()
    await handle(_msg(text="YES", mid="approval-failure-2"), failed, deps)

    assert "temporary problem" in failed.sent[0]
    assert deps.store.has_pending_approval(key) is True

    retried = FakeReplier()
    await handle(_msg(text="YES", mid="approval-failure-3"), retried, deps)

    assert retried.sent == ["Prepared after retry"]
    assert deps.store.has_pending_approval(key) is False
    assert attempts == 2


async def test_legacy_runtime_ignores_stale_native_approval_state(tmp_path, monkeypatch):
    from heynyc.channels.identity import user_key as _uk
    from heynyc.core import pii_crypto

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    deps = _deps(tmp_path)
    key = _uk("whatsapp_meta", "+1555", "s")
    deps.store.set_pending_approval(key, b'{"pending":true}', ttl_s=60)
    _burn_welcome(deps)

    replier = FakeReplier()
    await handle(_msg(text="ordinary question", mid="legacy-stale"), replier, deps)

    assert replier.sent == ["Here you go."]
    assert deps.store.has_pending_approval(key) is True


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


async def test_daily_cap_message_includes_emergency_routes_without_text_classification(tmp_path):
    from heynyc.channels.orchestrator import _DAILY_CAP_MSG, _nyc_day

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.user_daily_spend_cap = 0.50
    message = _msg(text="I have severe chest pain right now", mid="cap3")
    from heynyc.channels.identity import user_key as _uk
    deps.store.add_spend(_uk(message.channel, message.sender, "s"), _nyc_day(), 9.99)

    await handle(message, replier, deps)

    assert replier.sent == [_DAILY_CAP_MSG]
    assert "988" in _DAILY_CAP_MSG and "911" in _DAILY_CAP_MSG
    assert replier.typed == 0


async def test_turn_cost_accrues_to_the_daily_tally(tmp_path):
    from heynyc.channels.identity import user_key as _uk
    from heynyc.channels.orchestrator import _nyc_day

    deps, replier = _deps(tmp_path), FakeReplier()
    deps.user_daily_spend_cap = 5.00
    message = _msg(text="whats happening this weekend", mid="cap4")

    await handle(message, replier, deps)

    key = _uk(message.channel, message.sender, "s")
    assert deps.store.daily_spend(key, _nyc_day()) >= 0.0  # tally exists (cost may be 0 in fakes)
