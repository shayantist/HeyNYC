"""Session persistence, save/resume conversations as append-only JSONL.

Append-only JSONL (one message per line) is the proven, simple format used by
Claude Code et al.: cheap to append, easy to replay/rebuild on resume. A Session
wraps a Conversation and persists each committed turn.

At rest each line is encrypted PER RECORD with AES-256-GCM when `HEYNYC_PII_KEY`
is set (security-audit F1, see `pii_crypto`): a filled-form conversation carries
the user's name / SSN / DOB / address as chat text, so the transcript is a PII
sink. Per-line encryption keeps the append-only property (a fresh nonce per line)
while hiding the content; with no key the line stays cleartext (the insecure
dev/test path). The line format is fixed by the key state at read time: do not
flip the key against a file already written in the other mode. Retention is the
`purge_expired_sessions` TTL backstop.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import pii_crypto
from .agent import Agent, AgentResult, Conversation


def _encode_line(message: dict) -> str:
    """One transcript line: base64(AES-256-GCM token) when a key is set, else raw JSON."""
    raw = json.dumps(message)
    if pii_crypto.is_enabled():
        return base64.b64encode(pii_crypto.encrypt(raw)).decode("ascii")
    return raw


def _decode_line(line: str) -> dict:
    """Inverse of `_encode_line`. Decrypts (authenticated) when a key is set."""
    if pii_crypto.is_enabled():
        return json.loads(pii_crypto.decrypt(base64.b64decode(line)))
    return json.loads(line)


def purge_expired_sessions(sessions_dir: Path, max_age_days: float | None = None) -> list[str]:
    """Irreversibly delete session transcripts older than the retention window
    (default `HEYNYC_PII_RETENTION_DAYS`, else 30 days). The storage-limitation
    backstop (GDPR Art 5(1)(e)) for the transcript PII sink. Run by a cron/CLI:

        python -c "from heynyc.core.session import purge_expired_sessions; \\
                   purge_expired_sessions('.data/sessions')"
    """
    return pii_crypto.purge_expired_files(sessions_dir, "*.jsonl", max_age_days)


@dataclass
class Session:
    agent: Agent
    id: str
    path: Optional[Path] = None
    convo: Conversation = field(init=False)

    def __post_init__(self) -> None:
        self.convo = self.agent.conversation()

    @classmethod
    def load(cls, agent: Agent, session_id: str, path: Path) -> "Session":
        """Rebuild a session's history from its JSONL file (if present)."""
        session = cls(agent=agent, id=session_id, path=path)
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    session.convo.turns.append(_decode_line(line))
        return session

    def _append(self, *messages: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for message in messages:
                fh.write(_encode_line(message) + "\n")

    async def send(self, user_message: str, **kwargs) -> AgentResult:
        result = await self.convo.send(user_message, **kwargs)
        self._append(
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": result.text},
        )
        return result

    async def stream(self, user_message: str, **kwargs):
        """Stream a turn's events, persisting the committed turn afterward."""
        before = len(self.convo.turns)
        async for event in self.convo.stream(user_message, **kwargs):
            yield event
        self._append(*self.convo.turns[before:])

    @property
    def turns(self) -> list[dict]:
        return self.convo.turns
