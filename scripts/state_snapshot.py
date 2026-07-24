#!/usr/bin/env python3
"""Create, verify, and restore a portable HeyNYC resident-state snapshot."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

FORMAT_VERSION = 1
DATABASE = "channels.sqlite3"
SQLITE_SIDECARS = {f"{DATABASE}-shm", f"{DATABASE}-wal"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"resident state contains a symlink: {path}")
    if path.is_dir():
        for child in path.iterdir():
            _reject_symlinks(child)


def _reject_symlink_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError("restore target must not contain symlinks")


def _sqlite_user_version(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        result = db.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ValueError(f"SQLite integrity check failed: {result}")
        return int(db.execute("PRAGMA user_version").fetchone()[0])


def _manifest_files(data_dir: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(item for item in data_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def create_snapshot(
    data_dir: Path, output: Path, app_sha: str, *, quiesced: bool = False
) -> Path:
    """Snapshot resident state into a new, self-verifying directory."""
    if not quiesced:
        raise ValueError("snapshot requires an explicitly quiesced service")
    data_dir = Path(data_dir).resolve()
    output = Path(output).resolve()
    if len(app_sha) != 40 or any(char not in "0123456789abcdef" for char in app_sha.lower()):
        raise ValueError("app SHA must be a full 40-character hexadecimal commit")
    if output.exists():
        raise FileExistsError(f"snapshot target already exists: {output}")
    database = data_dir / DATABASE
    if not database.is_file() or database.is_symlink():
        raise FileNotFoundError(f"required SQLite state not found: {database}")
    _reject_symlinks(data_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot_data = temp / "data"
        snapshot_data.mkdir()
        with sqlite3.connect(database) as source, sqlite3.connect(
            snapshot_data / DATABASE
        ) as destination:
            source.backup(destination)
        for source in data_dir.iterdir():
            if source.name == DATABASE or source.name in SQLITE_SIDECARS:
                continue
            destination = snapshot_data / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise ValueError("data directory contains an unsupported special file")
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "app_sha": app_sha.lower(),
            "quiesced": True,
            "sqlite_user_version": _sqlite_user_version(snapshot_data / DATABASE),
            "files": _manifest_files(snapshot_data),
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def verify_snapshot(snapshot: Path) -> dict:
    """Verify the manifest, every file hash, and the SQLite database."""
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported snapshot format: {manifest.get('format_version')}")
    if manifest.get("quiesced") is not True:
        raise ValueError("snapshot manifest does not record a quiesced service")
    app_sha = manifest.get("app_sha")
    if not isinstance(app_sha, str) or len(app_sha) != 40 or any(
        char not in "0123456789abcdef" for char in app_sha
    ):
        raise ValueError("snapshot manifest app_sha is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("snapshot manifest files must be a list")
    data_dir = snapshot / "data"
    _reject_symlinks(data_dir)
    expected = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in entry["sha256"])
        ):
            raise ValueError("snapshot manifest file entry is invalid")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe snapshot path: {relative}")
        path = data_dir.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"snapshot file missing: {relative}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"size mismatch for {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"hash mismatch for {relative}")
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(data_dir).as_posix()
        for path in data_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("snapshot contains unmanifested files")
    version = _sqlite_user_version(data_dir / DATABASE)
    if version != manifest.get("sqlite_user_version"):
        raise ValueError("SQLite schema version does not match the manifest")
    return manifest


def _json_lines(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def _verify_application_state(data_dir: Path) -> None:
    """Prove the configured key and current application can open restored state."""
    from heynyc.channels.store import ChannelStore
    from heynyc.core import pii_crypto

    if not pii_crypto.is_enabled():
        raise pii_crypto.PiiCryptoError("HEYNYC_PII_KEY is required to verify restored state")
    store = ChannelStore(
        data_dir / DATABASE, rate_limit=1, window_s=1, dedup_ttl_s=1
    )
    store._db.close()
    with sqlite3.connect(data_dir / DATABASE) as db:
        for payload, outbox in db.execute("SELECT payload, outbox FROM inbox"):
            if payload is not None:
                json.loads(pii_crypto.decrypt(payload))
            if outbox is not None:
                json.loads(pii_crypto.decrypt(outbox))
        for (state,) in db.execute("SELECT state FROM approval_pending"):
            json.loads(pii_crypto.decrypt(state))
    for path in (data_dir / "sessions").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(pii_crypto.decrypt(base64.b64decode(line, validate=True)))
    for path in (data_dir / "drafts").glob("*.json"):
        json.loads(pii_crypto.decrypt(path.read_bytes()))
    feedback = data_dir / "feedback.jsonl"
    if feedback.exists():
        for line in feedback.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(pii_crypto.decrypt(base64.b64decode(line, validate=True)))
    _json_lines(data_dir / "telemetry.jsonl")
    _json_lines(data_dir / "outcomes.jsonl")


def restore_snapshot(snapshot: Path, target: Path) -> Path:
    """Restore a verified snapshot only into a missing or empty directory."""
    snapshot = Path(snapshot).resolve()
    target = Path(target).absolute()
    verify_snapshot(snapshot)
    _reject_symlink_path(target)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"restore target is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copytree(snapshot / "data", temp, dirs_exist_ok=True)
        _sqlite_user_version(temp / DATABASE)
        _verify_application_state(temp)
        if target.exists():
            target.rmdir()
        temp.rename(target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--data-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--app-sha", required=True)
    create.add_argument("--quiesced", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("snapshot", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("snapshot", type=Path)
    restore.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create_snapshot(args.data_dir, args.output, args.app_sha, quiesced=args.quiesced)
    elif args.command == "verify":
        verify_snapshot(args.snapshot)
    else:
        restore_snapshot(args.snapshot, args.target)
    print("ok")


if __name__ == "__main__":
    main()
