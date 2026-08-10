from __future__ import annotations

import base64
import importlib.util
import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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
    store.set_pending_approval("resident", b'{"pending":"approval"}', ttl_s=3600)
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
    snapshot.restore_snapshot(
        bundle,
        restored,
        deletion_generation_path=source / ".deletion-generation",
    )

    assert manifest["format_version"] == 2
    assert manifest["app_sha"] == "a" * 40
    assert manifest["sqlite_user_version"] == 4
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
    restored_store = ChannelStore(
        restored / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600
    )
    assert restored_store.pop_pending_approval("resident") == b'{"pending":"approval"}'
    restored_line = (restored / "sessions" / "resident.jsonl").read_text().strip()
    assert json.loads(pii_crypto.decrypt(base64.b64decode(restored_line)))["content"] == "hello"
    assert pii_crypto.deletion_generation(restored / ".deletion-generation") == 0


def test_snapshot_application_check_does_not_create_a_restore_copy(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )

    snapshot.verify_snapshot(bundle, application_state=True)

    assert {path.name for path in tmp_path.iterdir()} == {"source", "snapshot"}


def test_snapshot_application_check_does_not_migrate_the_snapshot(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )
    database = bundle / "data" / "channels.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA user_version = 3")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sqlite_user_version"] = 3
    database_entry = next(
        entry for entry in manifest["files"] if entry["path"] == "channels.sqlite3"
    )
    database_entry["size"] = database.stat().st_size
    database_entry["sha256"] = snapshot._sha256(database)
    manifest_path.write_text(json.dumps(manifest))
    before = snapshot._sha256(database)

    with pytest.raises(ValueError, match="schema version"):
        snapshot.verify_snapshot(bundle, application_state=True)

    assert snapshot._sha256(database) == before


def test_snapshot_application_check_rejects_a_malformed_key_without_records(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    source.mkdir()
    store = ChannelStore(
        source / "channels.sqlite3", rate_limit=1, window_s=1, dedup_ttl_s=1
    )
    store._db.close()
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )
    monkeypatch.setenv("HEYNYC_PII_KEY", "not-base64")

    with pytest.raises(pii_crypto.PiiCryptoError, match="valid base64"):
        snapshot.verify_snapshot(bundle, application_state=True)


def test_snapshot_records_but_does_not_copy_the_deletion_generation(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    generation = source / ".deletion-generation"
    pii_crypto.advance_deletion_generation(generation)

    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text())

    assert manifest["deletion_generation"] == 1
    assert not (bundle / "data" / generation.name).exists()


def test_restore_rejects_a_snapshot_from_before_confirmed_deletion(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    generation = source / ".deletion-generation"
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )
    pii_crypto.advance_deletion_generation(generation)
    target = tmp_path / "restored"

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "restore",
            bundle,
            "--target",
            target,
            "--deletion-generation",
            generation,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "predates a confirmed deletion" in result.stderr
    assert not target.exists()


def test_restore_api_requires_the_live_deletion_generation(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )

    with pytest.raises(TypeError, match="deletion_generation_path"):
        snapshot.restore_snapshot(bundle, tmp_path / "restored")


def test_restore_rejects_an_expired_snapshot(tmp_path: Path) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    _write_state(source)
    bundle = snapshot.create_snapshot(
        source,
        tmp_path / "snapshot",
        "a" * 40,
        quiesced=True,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="retention period"):
        snapshot.restore_snapshot(
            bundle,
            tmp_path / "restored",
            deletion_generation_path=source / ".deletion-generation",
        )


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
        snapshot.restore_snapshot(
            bundle,
            target,
            deletion_generation_path=source / ".deletion-generation",
        )

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
        snapshot.restore_snapshot(
            bundle,
            linked_target,
            deletion_generation_path=source / ".deletion-generation",
        )

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_target)
    with pytest.raises(ValueError, match="symlink"):
        snapshot.restore_snapshot(
            bundle,
            linked_parent / "child",
            deletion_generation_path=source / ".deletion-generation",
        )


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
        snapshot.restore_snapshot(
            bundle,
            target,
            deletion_generation_path=source / ".deletion-generation",
        )

    assert not target.exists()


def test_restore_rejects_wrong_key_for_pending_approval_only(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _module()
    source = tmp_path / "source"
    source.mkdir()
    store = ChannelStore(
        source / "channels.sqlite3", rate_limit=20, window_s=60, dedup_ttl_s=3600
    )
    store.set_pending_approval("resident", b'{"pending":"approval"}', ttl_s=3600)
    store._db.close()
    bundle = snapshot.create_snapshot(
        source, tmp_path / "snapshot", "3" * 40, quiesced=True
    )
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())

    with pytest.raises(pii_crypto.PiiCryptoError):
        target = tmp_path / "restored"
        snapshot.restore_snapshot(
            bundle,
            target,
            deletion_generation_path=source / ".deletion-generation",
        )

    assert not target.exists()
