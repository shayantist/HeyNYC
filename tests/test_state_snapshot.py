from __future__ import annotations

import base64
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from heynyc.channels.store import ChannelStore
from heynyc.core import pii_crypto

SCRIPT = Path(__file__).parents[1] / "scripts" / "state_snapshot.py"


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())


def _module():
    assert SCRIPT.exists(), "state snapshot script has not been implemented"
    spec = importlib.util.spec_from_file_location("state_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_state(data_dir: Path) -> None:
    data_dir.mkdir()
    store = ChannelStore(
        data_dir / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600
    )
    store.enqueue("SM123", "resident", json.dumps({"text": "encrypted inbound"}))
    store._db.close()
    (data_dir / "sessions").mkdir()
    session = base64.b64encode(
        pii_crypto.encrypt(json.dumps({"role": "user", "content": "hello"}))
    ).decode("ascii")
    (data_dir / "sessions" / "resident.jsonl").write_text(session + "\n")
    (data_dir / "drafts").mkdir()
    (data_dir / "drafts" / "resident.json").write_bytes(
        pii_crypto.encrypt(json.dumps({"snap": {"slots": {"borough": "Queens"}}}))
    )
    feedback = base64.b64encode(pii_crypto.encrypt(json.dumps({"flag": "wrong"}))).decode(
        "ascii"
    )
    (data_dir / "feedback.jsonl").write_text(feedback + "\n")
    (data_dir / "telemetry.jsonl").write_text(json.dumps({"turns": 1}) + "\n")
    (data_dir / "outcomes.jsonl").write_text(json.dumps({"form_ready": False}) + "\n")
    (data_dir / "index.lance").mkdir()
    (data_dir / "index.lance" / "rebuildable").write_text("not resident state")


def test_snapshot_verify_and_restore_round_trip(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)

    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "a" * 40, quiesced=True)
    manifest = snapshot.verify_snapshot(bundle)
    restored = tmp_path / "restored"
    snapshot.restore_snapshot(bundle, restored)

    assert manifest["format_version"] == 1
    assert manifest["app_sha"] == "a" * 40
    assert manifest["sqlite_user_version"] == 2
    assert {entry["path"] for entry in manifest["files"]} == {
        "channels.sqlite3",
        "drafts/resident.json",
        "feedback.jsonl",
        "outcomes.jsonl",
        "sessions/resident.jsonl",
        "telemetry.jsonl",
        "index.lance/rebuildable",
    }
    with sqlite3.connect(restored / "channels.sqlite3") as db:
        assert db.execute("SELECT message_id, state FROM inbox").fetchall() == [
            ("SM123", "received")
        ]
    restored_line = (restored / "sessions" / "resident.jsonl").read_text().strip()
    assert json.loads(pii_crypto.decrypt(base64.b64decode(restored_line)))["content"] == "hello"


def test_verify_rejects_tampered_file(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "b" * 40, quiesced=True)
    copied = bundle / "data" / "feedback.jsonl"
    copied.write_bytes(b"x" * copied.stat().st_size)

    with pytest.raises(ValueError, match="hash mismatch"):
        snapshot.verify_snapshot(bundle)


def test_verify_rejects_malformed_manifest_metadata(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "2" * 40, quiesced=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["app_sha"] = "not-a-commit"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="app_sha"):
        snapshot.verify_snapshot(bundle)

    manifest["app_sha"] = "2" * 40
    manifest["files"][0]["size"] = "large"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest file entry"):
        snapshot.verify_snapshot(bundle)


def test_restore_refuses_nonempty_target(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "c" * 40, quiesced=True)
    target = tmp_path / "target"
    _write_state(target)
    (target / "index.lance" / "rebuildable").write_text("keep this index")

    with pytest.raises(FileExistsError, match="not empty"):
        snapshot.restore_snapshot(bundle, target)

    assert (target / "index.lance" / "rebuildable").read_text() == "keep this index"
    assert json.loads((bundle / "manifest.json").read_text())["app_sha"] == "c" * 40


def test_snapshot_rejects_symlinked_resident_state(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    (source / "sessions" / "resident.jsonl").unlink()
    (source / "sessions" / "resident.jsonl").symlink_to(source / "feedback.jsonl")

    with pytest.raises(ValueError, match="symlink"):
        snapshot.create_snapshot(source, tmp_path / "snapshot", "d" * 40, quiesced=True)


def test_restore_rejects_a_symlink_target(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "e" * 40, quiesced=True)
    real_target = tmp_path / "real-target"
    real_target.mkdir()
    linked_target = tmp_path / "linked-target"
    linked_target.symlink_to(real_target)

    with pytest.raises(ValueError, match="symlink"):
        snapshot.restore_snapshot(bundle, linked_target)

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_target)
    with pytest.raises(ValueError, match="symlink"):
        snapshot.restore_snapshot(bundle, linked_parent / "child")


def test_snapshot_requires_explicit_quiescence(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)

    with pytest.raises(ValueError, match="quiesced"):
        snapshot.create_snapshot(source, tmp_path / "snapshot", "f" * 40)


def test_restore_fails_when_the_existing_key_cannot_decrypt_state(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(source, tmp_path / "snapshot", "1" * 40, quiesced=True)
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())

    with pytest.raises(pii_crypto.PiiCryptoError):
        target = tmp_path / "restored"
        snapshot.restore_snapshot(bundle, target)

    assert not target.exists()
