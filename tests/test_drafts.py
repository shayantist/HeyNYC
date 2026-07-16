"""The per-user structured draft store — real state, not LLM reconstruction."""
from __future__ import annotations

import os
import time

from heynyc.core import pii_crypto
from heynyc.core.drafts import DraftStore


def test_merge_accumulates_losslessly_across_turns(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"legal_name": "Ana Diaz"})
    merged = d.merge("snap", {"monthly_income": 1500})   # next turn passes only the new field
    assert merged == {"legal_name": "Ana Diaz", "monthly_income": 1500}   # turn-1 name retained


def test_edits_overwrite_but_empty_does_not_clobber(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"monthly_income": 1500})
    assert d.merge("snap", {"monthly_income": 1800})["monthly_income"] == 1800   # an edit
    assert d.merge("snap", {"monthly_income": ""})["monthly_income"] == 1800       # empty ignored


def test_persists_across_store_instances(tmp_path):
    DraftStore(tmp_path).for_user("u1").merge("snap", {"legal_name": "Ana"})
    # a fresh store (a later request / restart) sees the persisted draft
    assert DraftStore(tmp_path).for_user("u1").load("snap") == {"legal_name": "Ana"}


def test_users_are_isolated(tmp_path):
    store = DraftStore(tmp_path)
    store.for_user("u1").merge("snap", {"legal_name": "Ana"})
    assert store.for_user("u2").load("snap") == {}


def test_clear_removes_a_program_draft(tmp_path):
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"legal_name": "Ana"})
    d.clear("snap")
    assert d.load("snap") == {}


# --- Encryption at rest (security-audit F1) ---------------------------------


def test_round_trips_through_encryption_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"legal_name": "Ana Diaz", "ssn": "123-45-6789"})
    # a fresh store (later request / restart) transparently decrypts
    assert DraftStore(tmp_path).for_user("u1").load("snap") == {
        "legal_name": "Ana Diaz",
        "ssn": "123-45-6789",
    }


def test_on_disk_bytes_are_ciphertext_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    d = DraftStore(tmp_path).for_user("u1")
    d.merge("snap", {"ssn": "123-45-6789"})
    raw = (tmp_path / "u1.json").read_bytes()
    assert b"123-45-6789" not in raw  # the PII is not on disk in the clear
    assert b"ssn" not in raw


def test_stays_cleartext_when_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)  # the dev path
    DraftStore(tmp_path).for_user("u1").merge("snap", {"legal_name": "Ana"})
    raw = (tmp_path / "u1.json").read_text()
    assert "Ana" in raw  # unchanged, insecure-by-design dev behavior


def test_plaintext_draft_is_migrated_when_encryption_is_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)
    DraftStore(tmp_path).for_user("u1").merge("snap", {"legal_name": "Ana Diaz"})
    assert "Ana Diaz" in (tmp_path / "u1.json").read_text()

    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    assert DraftStore(tmp_path).migrate_plaintext() == [str(tmp_path / "u1.json")]
    assert b"Ana Diaz" not in (tmp_path / "u1.json").read_bytes()
    assert DraftStore(tmp_path).for_user("u1").load("snap") == {"legal_name": "Ana Diaz"}


def test_merge_fails_closed_on_malformed_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", "!!!not-base64!!!")
    d = DraftStore(tmp_path).for_user("u1")
    try:
        d.merge("snap", {"ssn": "123-45-6789"})
        assert False, "expected a fail-closed error, not a cleartext write"
    except pii_crypto.PiiCryptoError:
        pass
    # nothing (least of all cleartext PII) was persisted
    p = tmp_path / "u1.json"
    assert not p.exists() or b"123-45-6789" not in p.read_bytes()


# --- Retention / TTL sweep (irreversible; GDPR Art 5(1)(e)) -----------------


def _age_file(path, days: float) -> None:
    old = time.time() - days * 86400
    os.utime(path, (old, old))


def test_purge_expired_deletes_old_keeps_recent(tmp_path):
    store = DraftStore(tmp_path)
    store.for_user("old").merge("snap", {"legal_name": "Old"})
    store.for_user("new").merge("snap", {"legal_name": "New"})
    _age_file(tmp_path / "old.json", days=45)
    deleted = store.purge_expired(max_age_days=30)
    assert not (tmp_path / "old.json").exists()  # irreversibly gone
    assert (tmp_path / "new.json").exists()
    assert any("old.json" in p for p in deleted)


def test_purge_expired_empty_dir_is_noop(tmp_path):
    assert DraftStore(tmp_path / "missing").purge_expired(max_age_days=30) == []
