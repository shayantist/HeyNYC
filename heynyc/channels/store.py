"""Crash-aware, Redis-free dedup + rate-limit on stdlib sqlite3. Survives restarts;
the one residual loss window (a crash between the 200 and the reply) is acceptable
for the pilot and closes by swapping the dispatch() seam for a real queue."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class ChannelStore:
    def __init__(self, path: Path, *, rate_limit: int, window_s: int, dedup_ttl_s: int) -> None:
        self.rate_limit = rate_limit
        self.window_s = window_s
        self.dedup_ttl_s = dedup_ttl_s
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS seen "
            "(message_id TEXT PRIMARY KEY, ts REAL, user_key TEXT NOT NULL DEFAULT '')"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(seen)")}
        if "user_key" not in columns:
            self._db.execute(
                "ALTER TABLE seen ADD COLUMN user_key TEXT NOT NULL DEFAULT ''"
            )
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
        self._db.commit()

    def seen(self, message_id: str, user_key: str = "") -> bool:
        """True if already seen; otherwise record it and return False. INSERT OR IGNORE
        keeps the prune + insert in one committed transaction (no dangling lock)."""
        now = time.time()
        self._db.execute("DELETE FROM seen WHERE ts < ?", (now - self.dedup_ttl_s,))
        cur = self._db.execute(
            "INSERT OR IGNORE INTO seen (message_id, ts, user_key) VALUES (?, ?, ?)",
            (message_id, now, user_key),
        )
        self._db.commit()
        return cur.rowcount == 0

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
