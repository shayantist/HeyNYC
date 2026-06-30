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
        self._db.execute("CREATE TABLE IF NOT EXISTS seen (message_id TEXT PRIMARY KEY, ts REAL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS rate (user_key TEXT, ts REAL)")
        self._db.execute("CREATE INDEX IF NOT EXISTS rate_key ON rate (user_key, ts)")
        self._db.commit()

    def seen(self, message_id: str) -> bool:
        """True if already seen; otherwise record it and return False. INSERT OR IGNORE
        keeps the prune + insert in one committed transaction (no dangling lock)."""
        now = time.time()
        self._db.execute("DELETE FROM seen WHERE ts < ?", (now - self.dedup_ttl_s,))
        cur = self._db.execute(
            "INSERT OR IGNORE INTO seen (message_id, ts) VALUES (?, ?)", (message_id, now)
        )
        self._db.commit()
        return cur.rowcount == 0

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
