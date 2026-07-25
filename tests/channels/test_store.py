import sqlite3
import time

import pytest

from heynyc.channels.store import ChannelStore
from heynyc.core import pii_crypto


def _store(tmp_path, **kw):
    kw.setdefault("rate_limit", 3)
    kw.setdefault("window_s", 60)
    kw.setdefault("dedup_ttl_s", 3600)
    return ChannelStore(tmp_path / "ch.sqlite3", **kw)


def test_seen_dedups_and_persists(tmp_path):
    s = _store(tmp_path)
    assert s.seen("wamid.A") is False   # first time
    assert s.seen("wamid.A") is True    # repeat
    reopened = _store(tmp_path)
    assert reopened.seen("wamid.A") is True   # survives a restart


def test_old_seen_schema_migrates_into_the_single_inbox_dedup_table(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    path = tmp_path / "ch.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE seen "
        "(message_id TEXT PRIMARY KEY, ts REAL, user_key TEXT NOT NULL DEFAULT '')"
    )
    db.execute(
        "INSERT INTO seen (message_id, ts, user_key) VALUES (?, ?, ?)",
        ("SM-old", time.time(), "u1"),
    )
    db.commit()
    db.close()

    store = _store(tmp_path)

    assert store.enqueue("SM-old", "u1", '{"text":"duplicate"}') is False
    tables = {
        row[0]
        for row in store._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "inbox" in tables
    assert "seen" not in tables
    assert "approval_pending" in tables
    assert store._db.execute("PRAGMA user_version").fetchone()[0] == 4


def test_store_refuses_to_downgrade_a_newer_schema(tmp_path):
    path = tmp_path / "ch.sqlite3"
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 99")
    db.commit()
    db.close()

    with pytest.raises(RuntimeError, match="newer than supported"):
        _store(tmp_path)

    reopened = sqlite3.connect(path)
    assert reopened.execute("PRAGMA user_version").fetchone()[0] == 99


def test_inbox_claim_decrypts_payload_that_is_encrypted_at_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = _store(tmp_path)
    payload = '{"sender":"+15551234567","text":"I need help"}'

    assert store.enqueue("SM1", "u1", payload) is True

    stored = store._db.execute(
        "SELECT payload FROM inbox WHERE message_id = ?", ("SM1",)
    ).fetchone()[0]
    assert isinstance(stored, bytes)
    assert b"+15551234567" not in stored
    assert b"I need help" not in stored
    assert store.claim_next(lease_s=30) == {
        "message_id": "SM1",
        "user_key": "u1",
        "payload": payload,
        "attempts": 1,
        "outbox": None,
        "delivered_parts": 0,
    }


def test_inbox_completion_scrubs_payload_and_keeps_delivery_sids(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = _store(tmp_path)
    store.enqueue("SM-in", "u1", '{"text":"private"}')
    store.claim_next(lease_s=30)

    store.complete("SM-in", ["SM-out-1", "SM-out-2"])

    assert store._db.execute(
        "SELECT state, payload, outbound_ids FROM inbox WHERE message_id = ?", ("SM-in",)
    ).fetchone() == ("sent", None, '["SM-out-1", "SM-out-2"]')
    assert store.enqueue("SM-in", "u1", '{"text":"duplicate"}') is False


def test_inbox_claims_one_message_per_sender_in_arrival_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = _store(tmp_path)
    store.enqueue("u1-first", "u1", '{"text":"first"}')
    store.enqueue("u1-second", "u1", '{"text":"second"}')
    store.enqueue("u2-first", "u2", '{"text":"other"}')

    first = store.claim_next(user_key="u1", lease_s=30)

    assert first["message_id"] == "u1-first"
    assert store.claim_next(user_key="u1", lease_s=30) is None
    assert store.claim_next(user_key="u2", lease_s=30)["message_id"] == "u2-first"
    store.complete("u1-first", [])
    assert store.claim_next(user_key="u1", lease_s=30)["message_id"] == "u1-second"


def test_failed_inbox_message_waits_then_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    now = 100.0
    monkeypatch.setattr("heynyc.channels.store.time.time", lambda: now)
    store = _store(tmp_path)
    store.enqueue("SM1", "u1", '{"text":"retry me"}')
    assert store.claim_next(lease_s=30)["attempts"] == 1

    store.fail("SM1", retry_after_s=10)

    now = 109.0
    assert store.claim_next(lease_s=30) is None
    now = 110.0
    assert store.claim_next(lease_s=30)["attempts"] == 2


def test_expired_processing_lease_is_reclaimed_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    now = 100.0
    monkeypatch.setattr("heynyc.channels.store.time.time", lambda: now)
    store = _store(tmp_path)
    store.enqueue("SM-crash", "u1", '{"text":"recover me"}')
    assert store.claim_next(lease_s=10)["attempts"] == 1

    now = 109.0
    assert store.claim_next(lease_s=10) is None
    now = 110.0
    assert store.claim_next(lease_s=10)["attempts"] == 2


def test_terminal_failure_does_not_block_later_messages_from_same_sender(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    store = _store(tmp_path)
    store.enqueue("first", "u1", '{"text":"first"}')
    store.enqueue("second", "u1", '{"text":"second"}')
    assert store.claim_next(lease_s=30)["message_id"] == "first"

    store.fail("first")

    assert store.claim_next(lease_s=30)["message_id"] == "second"


def test_allow_trips_after_limit(tmp_path):
    s = _store(tmp_path, rate_limit=3, window_s=60)
    assert [s.allow("u1") for _ in range(4)] == [True, True, True, False]
    assert s.allow("u2") is True        # other users unaffected


def test_allow_window_resets(tmp_path):
    s = _store(tmp_path, rate_limit=1, window_s=1)
    assert s.allow("u1") is True
    assert s.allow("u1") is False
    time.sleep(1.1)
    assert s.allow("u1") is True


def test_pending_delete_stages_and_pops_once(tmp_path):
    s = _store(tmp_path)
    assert s.pop_pending_delete("u1") is None
    s.set_pending_delete("u1")
    s.set_pending_delete("u1")            # a fresh DELETE replaces the un-confirmed one
    staged = s.pop_pending_delete("u1")
    assert staged is not None
    assert s.pop_pending_delete("u1") is None   # consumed once


def test_pending_approval_is_encrypted_resident_bound_and_consumed_once(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    now = 100.0
    monkeypatch.setattr("heynyc.channels.store.time.time", lambda: now)
    s = _store(tmp_path)
    state = b'{"pending":{"tool":"prepare_snap_form","address":"private"}}'

    s.set_pending_approval("u1", state, ttl_s=60)

    stored = s._db.execute(
        "SELECT state FROM approval_pending WHERE user_key = ?", ("u1",)
    ).fetchone()[0]
    assert isinstance(stored, bytes)
    assert b"prepare_snap_form" not in stored
    assert b"private" not in stored
    assert s.has_pending_approval("u1") is True
    assert s.has_pending_approval("u2") is False
    assert s.pop_pending_approval("u2") is None
    assert s.pop_pending_approval("u1") == state
    assert s.pop_pending_approval("u1") is None


def test_pending_approval_ciphertext_cannot_move_between_residents(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    s = _store(tmp_path)
    s.set_pending_approval("u1", b'{"private":"resident one"}', ttl_s=60)
    encrypted = s._db.execute(
        "SELECT state FROM approval_pending WHERE user_key = ?",
        ("u1",),
    ).fetchone()[0]
    now = time.time()
    s._db.execute(
        "INSERT INTO approval_pending (user_key, state, ts, expires_at) VALUES (?, ?, ?, ?)",
        ("u2", encrypted, now, now + 60),
    )
    s._db.commit()

    with pytest.raises(pii_crypto.PiiCryptoError):
        s.pop_pending_approval("u2")


def test_pending_approval_rebinds_legacy_ciphertext_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    s = _store(tmp_path)
    now = time.time()
    s._db.execute(
        "INSERT INTO approval_pending "
        "(user_key, state, ts, expires_at, aad_bound) VALUES (?, ?, ?, ?, 0)",
        (
            "u1",
            pii_crypto.encrypt('{"legacy":"state"}'),
            now,
            now + 60,
        ),
    )
    s._db.commit()

    assert s.get_pending_approval("u1") == b'{"legacy":"state"}'
    state, aad_bound = s._db.execute(
        "SELECT state, aad_bound FROM approval_pending WHERE user_key = ?",
        ("u1",),
    ).fetchone()
    assert aad_bound == 1
    assert (
        pii_crypto.decrypt(
            state,
            associated_data=b"u1",
        )
        == '{"legacy":"state"}'
    )


def test_pending_approval_expires_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    now = 100.0
    monkeypatch.setattr("heynyc.channels.store.time.time", lambda: now)
    s = _store(tmp_path)
    s.set_pending_approval("u1", b"state", ttl_s=30)

    now = 131.0

    assert s.has_pending_approval("u1") is False
    assert s.pop_pending_approval("u1") is None
    assert s._db.execute(
        "SELECT 1 FROM approval_pending WHERE user_key = ?", ("u1",)
    ).fetchone() is None


def test_delete_user_removes_flags_and_inbox_but_keeps_spend(tmp_path, monkeypatch):
    """DELETE MY DATA wipes the resident's own flag rows (pending + confirmed) but leaves the
    anonymized daily spend record standing for abuse control (the survivor promised in the copy)."""
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    s = _store(tmp_path)
    s.set_pending_flag("u1", 3, "report")
    s.add_flag("u1", 5, "report")
    s.set_pending_delete("u1")
    s.set_pending_approval("u1", b"pending state", ttl_s=60)
    s.add_spend("u1", "2026-07-20", 0.07)
    assert s.first_contact("u1") is True
    assert s.allow("u1") is True
    s.add_flag("u2", 1, "report")         # another user is untouched
    s.enqueue("SM-u1", "u1", '{"text":"delete me"}')
    s.enqueue("SM-u2", "u2", '{"text":"keep me"}')

    s.delete_user("u1")

    remaining = s.flags()
    assert [f["user_key"] for f in remaining] == ["u2"]   # only the other user's flag is left
    assert s.pop_pending_flag("u1") is None
    assert s.pop_pending_delete("u1") is None
    assert s.pop_pending_approval("u1") is None
    assert abs(s.daily_spend("u1", "2026-07-20") - 0.07) < 1e-9   # spend survives
    assert s._db.execute(
        "SELECT message_id FROM inbox ORDER BY message_id"
    ).fetchall() == [("SM-u2",)]
    assert s._db.execute("SELECT 1 FROM welcomed WHERE user_key = 'u1'").fetchone() is None
    assert s._db.execute("SELECT 1 FROM rate WHERE user_key = 'u1'").fetchone() is None


def test_inbox_retention_purge_removes_expired_payloads_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    now = 100.0
    monkeypatch.setattr("heynyc.channels.store.time.time", lambda: now)
    store = _store(tmp_path)
    store.enqueue("old", "u1", '{"text":"old"}')
    now = 200.0
    store.enqueue("new", "u2", '{"text":"new"}')

    store.purge_inbox(before=150.0)

    assert store._db.execute(
        "SELECT message_id FROM inbox ORDER BY message_id"
    ).fetchall() == [("new",)]


def test_first_contact_is_true_once_then_false_forever(tmp_path):
    s = _store(tmp_path)
    assert s.first_contact("u1") is True
    assert s.first_contact("u1") is False
    assert s.first_contact("u2") is True          # per user
    reopened = _store(tmp_path)
    assert reopened.first_contact("u1") is False  # survives a restart (once EVER)
