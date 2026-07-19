"""User error-feedback loop (OTI Gap 6): a resident flags the last answer as wrong,
it lands in a PII-redacted, append-only log keyed off the salted user_key, and the owner
can aggregate the flags to spot a systematic error. All offline: no model, no network."""
import asyncio

from heynyc.channels import analytics
from heynyc.channels.base import InboundMessage, KeyedLocks
from heynyc.channels.identity import user_key
from heynyc.channels.orchestrator import Deps, flag_note, handle, is_flag
from heynyc.channels.store import ChannelStore
from heynyc.core import config, pii_crypto
from heynyc.core.agent import Agent
from heynyc.core.registry import Registry


# ---- PII redaction (write-time) ------------------------------------------------

def test_redact_pii_masks_phone_ssn_email_address_dob():
    note = ("the hours are wrong, call me at 212-555-1234 or "
            "reach me@example.com, ssn 123-45-6789, I live at 350 Jay Street, dob 03/04/1990")
    out = analytics.redact_pii(note)
    assert "212-555-1234" not in out
    assert "123-45-6789" not in out
    assert "me@example.com" not in out
    assert "350 Jay Street" not in out
    assert "03/04/1990" not in out
    # the actual complaint survives so the owner can still act on it
    assert "hours are wrong" in out


def test_redact_pii_masks_anumber_and_card():
    # immigration A-number (mixed-status households) + a benefit/debit card number (EBT is 19
    # digits): the identifiers the phone/SSN patterns miss but the red-team suite probes.
    out = analytics.redact_pii(
        "save my A-number A123456789 and my EBT card 1234 5678 9012 3456 7 for next time")
    assert "A123456789" not in out
    assert "1234 5678 9012 3456 7" not in out and "3456" not in out
    assert "save my" in out and "for next time" in out


def test_redact_pii_masks_passport_and_bank_account_like_ids():
    out = analytics.redact_pii(
        "passport X1234567 and bank account 123456789012"
    )
    assert "X1234567" not in out
    assert "123456789012" not in out


def test_redact_pii_keeps_ordinary_text():
    assert analytics.redact_pii("the cooling center hours are outdated") == \
        "the cooling center hours are outdated"
    assert analytics.redact_pii("") == ""
    # ordinary short numbers and an 'a' before a small number are NOT swept up
    assert analytics.redact_pii("apartment 350 has 2 rooms") == "apartment 350 has 2 rooms"


# ---- The recorder redacts free text at write time, keeps the user_key ----------

def test_record_feedback_redacts_note_and_query_and_keeps_user_key(tmp_path):
    path = tmp_path / "fb.jsonl"
    rec = analytics.record_feedback(
        path, user_key="abc123", channel="whatsapp_meta", message_id="m1", flag="/wrong",
        note="you're wrong, my number is 212-555-1234",
        user_query="is 350 Jay Street rent stabilized?",
        agent_text="You said 350 Jay Street. Call 311 at 212-639-9675. {cite:S1}",
    )
    written = analytics.load_feedback(path)[0]
    assert written["user_key"] == "abc123"
    assert written["flag"] == "/wrong"
    # resident-authored free text is redacted at write time
    assert "212-555-1234" not in written["note"]
    assert "350 Jay Street" not in written["user_query"]
    # assistant text can echo resident PII, so it is redacted before persistence too
    assert "350 Jay Street" not in written["agent_text"]
    assert "212-639-9675" not in written["agent_text"]
    assert "Call 311" in written["agent_text"]
    assert rec["note"] == written["note"]


def test_feedback_is_encrypted_when_hosted_key_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    path = tmp_path / "fb.jsonl"
    analytics.record_feedback(
        path, user_key="abc123", channel="whatsapp_meta", message_id="m1", flag="wrong",
        note="Ana Diaz is undocumented", user_query="Ana Diaz needs SNAP", agent_text="Hi Ana",
    )

    assert b"Ana Diaz" not in path.read_bytes()
    assert analytics.load_feedback(path)[0]["note"] == "Ana Diaz is undocumented"


def test_plaintext_feedback_is_migrated_when_encryption_is_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)
    path = tmp_path / "fb.jsonl"
    analytics.record_feedback(
        path, user_key="abc123", channel="whatsapp_meta", message_id="m1", flag="wrong",
        note="Ana Diaz is undocumented", user_query="question", agent_text="answer",
    )
    assert "Ana Diaz" in path.read_text()

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    assert analytics.migrate_plaintext_feedback(path) is True
    assert b"Ana Diaz" not in path.read_bytes()
    assert analytics.load_feedback(path)[0]["note"] == "Ana Diaz is undocumented"


# ---- Aggregation report --------------------------------------------------------

def test_summarize_feedback_aggregates(tmp_path):
    path = tmp_path / "fb.jsonl"
    for i, (uk, flag, q) in enumerate([
        ("u1", "/wrong", "cooling center hours"),
        ("u2", "wrong", "cooling center hours"),
        ("u1", "report", "snap income limit"),
    ]):
        analytics.record_feedback(
            path, user_key=uk, channel="whatsapp_meta", message_id=f"m{i}",
            flag=flag, note="", user_query=q, agent_text="an answer",
        )
    summary = analytics.summarize_feedback(analytics.load_feedback(path))
    assert summary["total"] == 3
    assert summary["users"] == 2
    assert summary["by_flag"]["/wrong"] == 1 and summary["by_flag"]["wrong"] == 1
    # the repeat-flagged query is surfaced so the owner can spot the systematic error
    top = dict(summary["top_queries"])
    assert top["cooling center hours"] == 2


def test_summarize_feedback_empty():
    s = analytics.summarize_feedback([])
    assert s["total"] == 0 and s["users"] == 0 and s["top_queries"] == []


# ---- Orchestrator: recognize the command + parse the optional note -------------

def test_is_flag_recognizes_slash_command_and_bare_tokens():
    assert is_flag("/wrong") and is_flag("/wrong the hours are outdated")
    assert is_flag("wrong") and is_flag("  Report ") and is_flag("👎")
    assert not is_flag("what's wrong with my application?")


def test_flag_note_extracts_optional_note():
    assert flag_note("/wrong the hours are outdated") == "the hours are outdated"
    assert flag_note("/wrong") == ""
    assert flag_note("wrong") == ""      # bare token carries no note
    assert flag_note("👎") == ""


# ---- Orchestrator end-to-end: short-circuit, fixed ack, redacted log -----------

class FakeReplier:
    def __init__(self):
        self.sent, self.typed = [], 0

    async def send_text(self, text):
        self.sent.append(text)

    async def indicate_typing(self):
        self.typed += 1


def _agent(reply="Here you go."):
    async def complete_fn(messages, tool_schemas):
        return {"role": "assistant", "content": reply, "tool_calls": None}
    return Agent(Registry.discover(config.MODULES_DIR), tools={}, complete_fn=complete_fn, model="fake")


def _deps(tmp_path):
    store = ChannelStore(tmp_path / "ch.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600)
    return Deps(agent=_agent(), store=store, sessions_dir=tmp_path / "sessions", salt="s",
                telemetry_path=tmp_path / "t.jsonl", feedback_path=tmp_path / "fb.jsonl",
                locks=KeyedLocks(), semaphore=asyncio.Semaphore(8))


def _msg(text, mid):
    return InboundMessage(channel="whatsapp_meta", sender="+1555", text=text, message_id=mid)


async def test_report_then_yes_flags_one_exchange_and_stays_pii_free(tmp_path, monkeypatch):
    """The consent-gated flag: REPORT offers a confirmation, YES writes a content-free POINTER
    (user_key + turn position) into the channel store and the redacted aggregate record. Scope is
    exactly the last exchange, and the raw sender / phone never touch the store or the log."""
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    deps = _deps(tmp_path)
    await handle(_msg("when do cooling centers open?", "q1"), FakeReplier(), deps)

    # 1) REPORT: consent copy only, nothing recorded
    r1 = FakeReplier()
    await handle(_msg("/wrong the hours are stale, call me at 212-555-1234", "f1"), r1, deps)
    assert r1.typed == 0
    assert "yes" in r1.sent[0].lower()
    assert not (tmp_path / "fb.jsonl").exists() and deps.store.flags() == []

    # 2) YES: the flag is written
    r2 = FakeReplier()
    await handle(_msg("yes", "f2"), r2, deps)
    assert r2.sent == ["Sent. A human will review that one exchange."]

    key = user_key("whatsapp_meta", "+1555", "s")
    flags = deps.store.flags()
    assert len(flags) == 1 and flags[0]["user_key"] == key      # keyed off the salted user_key

    # the pointer references EXACTLY the last exchange (one user turn + one assistant turn)
    from heynyc.core.session import _decode_line

    turns = [
        _decode_line(line)
        for line in (tmp_path / "sessions" / f"{key}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    idx = flags[0]["turn_index"]
    assert turns[idx]["role"] == "assistant"
    assert turns[idx - 1]["role"] == "user" and turns[idx - 1]["content"] == "when do cooling centers open?"

    # the store holds a pointer only: no raw sender, no phone, no free-text reason
    store_bytes = (tmp_path / "ch.sqlite3").read_bytes()
    assert b"+1555" not in store_bytes and b"212-555-1234" not in store_bytes
    assert b"hours are stale" not in store_bytes
    # the redacted aggregate record was still written, without the pre-consent free-text note
    rec = analytics.load_feedback(tmp_path / "fb.jsonl")[0]
    assert rec["user_query"] == "when do cooling centers open?" and rec["note"] == ""
    assert "212-555-1234" not in (tmp_path / "fb.jsonl").read_text()


async def test_flag_with_nothing_to_flag_yet(tmp_path):
    deps = _deps(tmp_path)
    r = FakeReplier()
    await handle(_msg("/wrong", "f0"), r, deps)     # no prior turn, nothing to confirm
    assert r.typed == 0
    assert not (tmp_path / "fb.jsonl").exists() and deps.store.flags() == []
    assert "ask me something" in r.sent[0].lower()


def test_report_command_is_advertised_for_discovery():
    """A resident must be able to find the command: the PRIVACY story mentions it, and so does the
    onboarding/help copy that also points at NEW."""
    from heynyc.channels.orchestrator import _privacy_message

    assert "report" in _privacy_message("whatsapp_meta").lower()
    welcome = Registry.discover(config.MODULES_DIR).welcome_text().lower()
    assert "report" in welcome
