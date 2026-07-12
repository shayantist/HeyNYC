"""Per-user structured draft store — the agent's *actual* memory for an in-progress form.

The problem it solves: conversation history is persisted (Session JSONL), but the form's
answers live as chat text the LLM re-derives each turn — lossy for a long, legal form. This
store persists the **validated slot dict** as structured JSON keyed by (user_key, program),
so the agent reads state instead of reconstructing it. It is the seed of the once-only vault:
SNAP-collected answers can later pre-fill other programs.

Privacy: keyed by the salted `user_key` (no raw identifiers); holds in-progress PII.
At rest it is encrypted with AES-256-GCM when `HEYNYC_PII_KEY` is set (security-audit
F1, see `pii_crypto`); with no key it stays cleartext, the insecure dev/test path.
Retention is bounded: `clear()` on flow completion, and `purge_expired()` as the
irreversible TTL backstop. One JSON file per user, mirroring the Session JSONL pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import pii_crypto


class UserDrafts:
    """A draft accessor already bound to one user (the channel creates it with the user_key
    baked in, so the engine never has to know the identity)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict:
        try:
            raw = self._path.read_bytes()
        except (FileNotFoundError, OSError):
            return {}
        if not raw:
            return {}
        if pii_crypto.is_enabled():
            blob = pii_crypto.decrypt(raw)  # authenticated; PiiCryptoError propagates (fail closed)
        else:
            try:
                blob = raw.decode("utf-8")
            except ValueError:
                return {}
        try:
            return json.loads(blob)
        except ValueError:
            return {}

    def _write(self, data: dict) -> None:
        """Persist the draft, encrypted at rest when a key is set (else cleartext dev path)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(data)
        if pii_crypto.is_enabled():
            self._path.write_bytes(pii_crypto.encrypt(blob))  # nonce || ciphertext || tag
        else:
            self._path.write_text(blob)

    def load(self, program: str) -> dict:
        """The current structured slots for this program (empty if no draft yet)."""
        return dict(self._read().get(program, {}).get("slots", {}))

    def merge(self, program: str, new_slots: dict) -> dict:
        """Fold this turn's answers into the persisted draft and return the full merged slots.
        Empty values don't clobber existing ones; non-empty values overwrite (an edit)."""
        data = self._read()
        slots = dict(data.get(program, {}).get("slots", {}))
        slots.update({k: v for k, v in (new_slots or {}).items() if v not in (None, "")})
        data[program] = {"slots": slots}
        self._write(data)
        return slots

    def clear(self, program: str) -> None:
        """Drop a program's draft. This is the flow-completion hook: once a filled
        application is produced, the caller should `clear()` so the PII does not
        linger to its TTL (see `DraftStore.purge_expired`)."""
        data = self._read()
        if program in data:
            del data[program]
            self._write(data)


class DraftStore:
    """JSON-file-per-user draft store. `for_user(key)` returns a bound `UserDrafts`."""

    def __init__(self, drafts_dir: Path) -> None:
        self._dir = Path(drafts_dir)

    def for_user(self, user_key: str) -> UserDrafts:
        return UserDrafts(self._dir / f"{user_key}.json")

    def purge_expired(self, max_age_days: float | None = None) -> list[str]:
        """Irreversibly delete draft files older than the retention window (default
        from `HEYNYC_PII_KEY`'s sibling `HEYNYC_PII_RETENTION_DAYS`, else 30 days).

        The storage-limitation backstop (GDPR Art 5(1)(e)) for any draft a
        completion `clear()` missed. Meant to be run by a cron/CLI, e.g.:

            python -c "from heynyc.core.drafts import DraftStore; \\
                       DraftStore('.data/drafts').purge_expired()"
        """
        return pii_crypto.purge_expired_files(self._dir, "*.json", max_age_days)
