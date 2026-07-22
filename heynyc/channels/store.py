"""Crash-aware inbox, dedup, and rate limits on stdlib sqlite3."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from heynyc.core import pii_crypto


class InboxPayloadError(RuntimeError):
    def __init__(self, message_id: str) -> None:
        super().__init__(f"unreadable inbox payload for {message_id}")
        self.message_id = message_id


class ChannelStore:
    def __init__(self, path: Path, *, rate_limit: int, window_s: int, dedup_ttl_s: int) -> None:
        self.rate_limit = rate_limit
        self.window_s = window_s
        self.dedup_ttl_s = dedup_ttl_s
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        tables = {
            row[0]
            for row in self._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "seen" in tables and "inbox" not in tables:
            self._db.execute("ALTER TABLE seen RENAME TO inbox")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS inbox "
            "(message_id TEXT PRIMARY KEY, ts REAL, user_key TEXT NOT NULL DEFAULT '')"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(inbox)")}
        if "user_key" not in columns:
            self._db.execute(
                "ALTER TABLE inbox ADD COLUMN user_key TEXT NOT NULL DEFAULT ''"
            )
        additions = {
            "payload": "BLOB",
            "outbox": "BLOB",
            "state": "TEXT NOT NULL DEFAULT 'sent'",
            "attempts": "INTEGER NOT NULL DEFAULT 0",
            "delivered_parts": "INTEGER NOT NULL DEFAULT 0",
            "available_at": "REAL NOT NULL DEFAULT 0",
            "lease_until": "REAL",
            "outbound_ids": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "REAL NOT NULL DEFAULT 0",
        }
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(inbox)")}
        for name, declaration in additions.items():
            if name not in columns:
                self._db.execute(f"ALTER TABLE inbox ADD COLUMN {name} {declaration}")
        self._db.execute("PRAGMA user_version = 2")
        self._db.execute("CREATE TABLE IF NOT EXISTS rate (user_key TEXT, ts REAL)")
        self._db.execute("CREATE INDEX IF NOT EXISTS rate_key ON rate (user_key, ts)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS spend "
            "(user_key TEXT, day TEXT, spent_usd REAL, PRIMARY KEY (user_key, day))"
        )
        # A confirmed flag is a POINTER, not a copy: (user_key -> session file) + turn_index
        # (position of the flagged assistant turn). The encrypted session JSONL holds the turns;
        # `heynyc feedback` joins on user_key and decrypts locally. No message content lives here.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS flag "
            "(user_key TEXT NOT NULL, turn_index INTEGER NOT NULL, flag TEXT NOT NULL DEFAULT '', "
            "ts REAL NOT NULL)"
        )
        # Consent gate: a REPORT stages one pending pointer per user until they reply YES.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS flag_pending "
            "(user_key TEXT PRIMARY KEY, turn_index INTEGER NOT NULL, flag TEXT NOT NULL DEFAULT '', "
            "ts REAL NOT NULL)"
        )
        # Consent gate for DELETE MY DATA: one pending deletion per user until they reply YES.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS delete_pending (user_key TEXT PRIMARY KEY, ts REAL NOT NULL)"
        )
        # First-contact marker: a durable once-EVER flag so a never-seen user gets the welcome
        # footer exactly once (the `seen` table is TTL-pruned, so it can't answer "ever").
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS welcomed (user_key TEXT PRIMARY KEY, ts REAL NOT NULL)"
        )
        self._db.commit()

    def seen(self, message_id: str, user_key: str = "") -> bool:
        """True if already seen; otherwise record it and return False. INSERT OR IGNORE
        keeps the prune + insert in one committed transaction (no dangling lock)."""
        now = time.time()
        self._db.execute(
            "DELETE FROM inbox WHERE state = 'sent' AND ts < ?", (now - self.dedup_ttl_s,)
        )
        cur = self._db.execute(
            "INSERT OR IGNORE INTO inbox (message_id, ts, user_key) VALUES (?, ?, ?)",
            (message_id, now, user_key),
        )
        self._db.commit()
        return cur.rowcount == 0

    def enqueue(self, message_id: str, user_key: str, payload: str) -> bool:
        """Persist one encrypted inbound message. False means this ID already exists."""
        now = time.time()
        self._db.execute(
            "DELETE FROM inbox WHERE state = 'sent' AND ts < ?", (now - self.dedup_ttl_s,)
        )
        cur = self._db.execute(
            "INSERT OR IGNORE INTO inbox "
            "(message_id, ts, user_key, payload, state, available_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'received', ?, ?)",
            (message_id, now, user_key, pii_crypto.encrypt(payload), now, now),
        )
        self._db.commit()
        return cur.rowcount == 1

    def claim_next(self, *, lease_s: float, user_key: str | None = None) -> dict | None:
        """Lease the oldest ready message, including work abandoned by a crashed worker."""
        now = time.time()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT i.message_id, i.user_key, i.payload, i.attempts, i.outbox, "
                "i.delivered_parts FROM inbox AS i "
                "WHERE i.payload IS NOT NULL AND (? IS NULL OR i.user_key = ?) AND "
                "NOT EXISTS (SELECT 1 FROM inbox AS earlier "
                "WHERE earlier.user_key = i.user_key AND earlier.payload IS NOT NULL AND "
                "earlier.state != 'failed' AND "
                "(earlier.ts < i.ts OR (earlier.ts = i.ts AND earlier.rowid < i.rowid))) AND "
                "i.available_at <= ? AND (i.state IN ('received', 'retrying') OR "
                "(i.state IN ('processing', 'delivering') AND COALESCE(i.lease_until, 0) <= ?)) "
                "ORDER BY i.ts, i.rowid LIMIT 1",
                (user_key, user_key, now, now),
            ).fetchone()
            if row is None:
                self._db.commit()
                return None
            attempts = int(row[3]) + 1
            self._db.execute(
                "UPDATE inbox SET state = 'processing', attempts = ?, lease_until = ?, "
                "updated_at = ? WHERE message_id = ?",
                (attempts, now + lease_s, now, row[0]),
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        try:
            payload = pii_crypto.decrypt(row[2])
            outbox = json.loads(pii_crypto.decrypt(row[4])) if row[4] is not None else None
        except Exception as exc:
            self.fail(row[0])
            raise InboxPayloadError(row[0]) from exc
        return {
            "message_id": row[0], "user_key": row[1], "payload": payload,
            "attempts": attempts, "outbox": outbox, "delivered_parts": int(row[5]),
        }

    def stage_outbox(self, message_id: str, parts: list[dict]) -> None:
        """Persist the complete rendered reply before the session commits or delivery starts."""
        self._db.execute(
            "UPDATE inbox SET outbox = ?, state = 'delivering', delivered_parts = 0, "
            "updated_at = ? WHERE message_id = ?",
            (pii_crypto.encrypt(json.dumps(parts)), time.time(), message_id),
        )
        self._db.commit()

    def complete(self, message_id: str, outbound_ids: list[str] | None = None) -> None:
        """Mark delivery accepted and erase the resident-authored queued payload."""
        now = time.time()
        if outbound_ids is None:
            self._db.execute(
                "UPDATE inbox SET state = 'sent', payload = NULL, outbox = NULL, "
                "lease_until = NULL, updated_at = ? WHERE message_id = ?",
                (now, message_id),
            )
        else:
            self._db.execute(
                "UPDATE inbox SET state = 'sent', payload = NULL, outbox = NULL, "
                "outbound_ids = ?, lease_until = NULL, updated_at = ? WHERE message_id = ?",
                (json.dumps(outbound_ids), now, message_id),
            )
        self._db.commit()

    def record_outbound(self, message_id: str, outbound_id: str) -> None:
        """Checkpoint one provider-accepted outbound message before sending the next part."""
        row = self._db.execute(
            "SELECT outbound_ids FROM inbox WHERE message_id = ?", (message_id,)
        ).fetchone()
        if row is None:
            return
        outbound_ids = json.loads(row[0] or "[]")
        if outbound_id not in outbound_ids:
            outbound_ids.append(outbound_id)
            self._db.execute(
                "UPDATE inbox SET outbound_ids = ?, updated_at = ? WHERE message_id = ?",
                (json.dumps(outbound_ids), time.time(), message_id),
            )
            self._db.execute(
                "UPDATE inbox SET delivered_parts = delivered_parts + 1 WHERE message_id = ?",
                (message_id,),
            )
            self._db.commit()

    def fail(self, message_id: str, *, retry_after_s: float | None = None) -> None:
        """Keep failed work encrypted, optionally releasing it for one later retry."""
        now = time.time()
        state = "retrying" if retry_after_s is not None else "failed"
        self._db.execute(
            "UPDATE inbox SET state = ?, available_at = ?, lease_until = NULL, "
            "updated_at = ? WHERE message_id = ?",
            (state, now + (retry_after_s or 0), now, message_id),
        )
        self._db.commit()

    def daily_spend(self, user_key: str, day: str) -> float:
        """This user's accumulated model cost for `day` (an ISO date string)."""
        row = self._db.execute(
            "SELECT spent_usd FROM spend WHERE user_key = ? AND day = ?", (user_key, day)
        ).fetchone()
        return float(row[0]) if row else 0.0

    def add_spend(self, user_key: str, day: str, amount: float) -> None:
        """Accrue one turn's model cost to the user's daily tally (upsert)."""
        self._db.execute(
            "INSERT INTO spend (user_key, day, spent_usd) VALUES (?, ?, ?) "
            "ON CONFLICT(user_key, day) DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd",
            (user_key, day, float(amount)),
        )
        self._db.commit()

    def set_pending_flag(self, user_key: str, turn_index: int, flag: str = "") -> None:
        """Stage a flag awaiting the resident's YES (one pending per user; a fresh REPORT replaces
        an un-confirmed one). Pointer only: turn position + a bounded command token, no free text."""
        self._db.execute(
            "INSERT INTO flag_pending (user_key, turn_index, flag, ts) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_key) DO UPDATE SET "
            "turn_index = excluded.turn_index, flag = excluded.flag, ts = excluded.ts",
            (user_key, int(turn_index), flag, time.time()),
        )
        self._db.commit()

    def pop_pending_flag(self, user_key: str) -> dict | None:
        """Read and clear this user's staged flag (consume-once). None if nothing is staged."""
        row = self._db.execute(
            "SELECT turn_index, flag, ts FROM flag_pending WHERE user_key = ?", (user_key,)
        ).fetchone()
        if row is None:
            return None
        self._db.execute("DELETE FROM flag_pending WHERE user_key = ?", (user_key,))
        self._db.commit()
        return {"turn_index": int(row[0]), "flag": row[1], "ts": float(row[2])}

    def add_flag(self, user_key: str, turn_index: int, flag: str = "") -> None:
        """Record a confirmed pointer to a flagged exchange (append-only). No message content."""
        self._db.execute(
            "INSERT INTO flag (user_key, turn_index, flag, ts) VALUES (?, ?, ?, ?)",
            (user_key, int(turn_index), flag, time.time()),
        )
        self._db.commit()

    def set_pending_delete(self, user_key: str) -> None:
        """Stage a DELETE MY DATA awaiting the resident's YES (one pending per user; a fresh
        DELETE replaces an un-confirmed one). No content: the presence of a row is the whole state."""
        self._db.execute(
            "INSERT INTO delete_pending (user_key, ts) VALUES (?, ?) "
            "ON CONFLICT(user_key) DO UPDATE SET ts = excluded.ts",
            (user_key, time.time()),
        )
        self._db.commit()

    def pop_pending_delete(self, user_key: str) -> dict | None:
        """Read and clear this user's staged deletion (consume-once). None if nothing is staged."""
        row = self._db.execute(
            "SELECT ts FROM delete_pending WHERE user_key = ?", (user_key,)
        ).fetchone()
        if row is None:
            return None
        self._db.execute("DELETE FROM delete_pending WHERE user_key = ?", (user_key,))
        self._db.commit()
        return {"ts": float(row[0])}

    def delete_user(self, user_key: str) -> None:
        """Erase this resident's inbox and control-plane rows on DELETE MY DATA. The daily
        `spend` record stays as the anonymized abuse-control survivor promised in the copy."""
        self._db.execute("DELETE FROM inbox WHERE user_key = ?", (user_key,))
        self._db.execute("DELETE FROM flag WHERE user_key = ?", (user_key,))
        self._db.execute("DELETE FROM flag_pending WHERE user_key = ?", (user_key,))
        self._db.execute("DELETE FROM delete_pending WHERE user_key = ?", (user_key,))
        self._db.execute("DELETE FROM welcomed WHERE user_key = ?", (user_key,))
        self._db.execute("DELETE FROM rate WHERE user_key = ?", (user_key,))
        self._db.commit()

    def purge_inbox(self, *, before: float) -> None:
        """Remove queued resident content older than the configured retention boundary."""
        self._db.execute("DELETE FROM inbox WHERE ts < ?", (before,))
        self._db.commit()

    def first_contact(self, user_key: str) -> bool:
        """True exactly once per user, EVER (the first call), then False. Marks and returns
        atomically so the first-contact welcome footer fires once and never again."""
        cur = self._db.execute(
            "INSERT OR IGNORE INTO welcomed (user_key, ts) VALUES (?, ?)", (user_key, time.time())
        )
        self._db.commit()
        return cur.rowcount == 1

    def flags(self) -> list[dict]:
        """Confirmed flag pointers, newest first, for the owner's `heynyc feedback` triage view."""
        rows = self._db.execute(
            "SELECT user_key, turn_index, flag, ts FROM flag ORDER BY ts DESC"
        ).fetchall()
        return [
            {"user_key": r[0], "turn_index": int(r[1]), "flag": r[2], "ts": float(r[3])}
            for r in rows
        ]

    def allow(self, user_key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_s
        self._db.execute("DELETE FROM rate WHERE ts < ?", (cutoff,))
        (count,) = self._db.execute(
            "SELECT COUNT(*) FROM rate WHERE user_key = ? AND ts >= ?", (user_key, cutoff)
        ).fetchone()
        if count >= self.rate_limit:
            self._db.commit()
            return False
        self._db.execute("INSERT INTO rate (user_key, ts) VALUES (?, ?)", (user_key, now))
        self._db.commit()
        return True
