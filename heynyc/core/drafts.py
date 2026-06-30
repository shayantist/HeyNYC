"""Per-user structured draft store — the agent's *actual* memory for an in-progress form.

The problem it solves: conversation history is persisted (Session JSONL), but the form's
answers live as chat text the LLM re-derives each turn — lossy for a long, legal form. This
store persists the **validated slot dict** as structured JSON keyed by (user_key, program),
so the agent reads state instead of reconstructing it. It is the seed of the once-only vault:
SNAP-collected answers can later pre-fill other programs.

Privacy: keyed by the salted `user_key` (no raw identifiers); holds in-progress PII
transiently; never logged. One JSON file per user, mirroring the Session JSONL pattern.
"""
from __future__ import annotations

import json
from pathlib import Path


class UserDrafts:
    """A draft accessor already bound to one user (the channel creates it with the user_key
    baked in, so the engine never has to know the identity)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict:
        try:
            return json.loads(self._path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return {}

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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data))
        return slots

    def clear(self, program: str) -> None:
        data = self._read()
        if program in data:
            del data[program]
            self._path.write_text(json.dumps(data))


class DraftStore:
    """JSON-file-per-user draft store. `for_user(key)` returns a bound `UserDrafts`."""

    def __init__(self, drafts_dir: Path) -> None:
        self._dir = Path(drafts_dir)

    def for_user(self, user_key: str) -> UserDrafts:
        return UserDrafts(self._dir / f"{user_key}.json")
