"""Session persistence, save/resume conversations as append-only JSONL.

Append-only JSONL (one message per line) is the proven, simple format used by
Claude Code et al.: cheap to append, easy to replay/rebuild on resume. A Session
wraps a Conversation and persists each committed turn.

At rest each line is encrypted PER RECORD with AES-256-GCM when `HEYNYC_PII_KEY`
is set (security-audit F1, see `pii_crypto`): a filled-form conversation carries
the user's name / SSN / DOB / address as chat text, so the transcript is a PII
sink. Per-line encryption keeps the append-only property (a fresh nonce per line)
while hiding the content; with no key the line stays cleartext (the insecure
dev/test path). Hosted startup migrates valid legacy cleartext lines before
accepting traffic. Retention is the `purge_expired_sessions` TTL backstop.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import pii_crypto
from .agent import AgentResult, turn_timestamp
from .citations import used_citations
from .memory import (
    ContextCapacityError,
    ContinuityRecord,
    continuity_reminder,
    merge_memory_usage,
)


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


def _migrate_plaintext_file(path: Path) -> bool:
    """Encrypt legacy JSON lines in one transcript while preserving encrypted lines."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    migrated = False
    encoded: list[str] = []
    for line in lines:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _decode_line(line)
            encoded.append(line)
        else:
            encoded.append(_encode_line(message))
            migrated = True
    if migrated:
        replacement = path.with_suffix(path.suffix + ".tmp")
        replacement.write_text("\n".join(encoded) + ("\n" if encoded else ""))
        replacement.replace(path)
    return migrated


def migrate_plaintext_sessions(sessions_dir: Path) -> list[str]:
    """Encrypt valid legacy cleartext transcripts before a hosted service accepts traffic."""
    if not pii_crypto.is_enabled():
        return []
    directory = Path(sessions_dir)
    if not directory.exists():
        return []
    return [str(path) for path in sorted(directory.glob("*.jsonl")) if _migrate_plaintext_file(path)]


def purge_expired_sessions(sessions_dir: Path, max_age_days: float | None = None) -> list[str]:
    """Irreversibly delete session transcripts older than the retention window
    (default `HEYNYC_PII_RETENTION_DAYS`, else 30 days). The storage-limitation
    backstop (GDPR Art 5(1)(e)) for the transcript PII sink. Run by a cron/CLI:

        python -c "from heynyc.core.session import purge_expired_sessions; \\
                   purge_expired_sessions('.data/sessions')"
    """
    return pii_crypto.purge_expired_files(sessions_dir, "*.jsonl", max_age_days)


@dataclass
class PendingTurn:
    user_message: str
    result: AgentResult
    continuity: ContinuityRecord | None = None
    runtime_state: bytes | None = None


@dataclass
class Session:
    agent: Any
    id: str
    path: Optional[Path] = None
    convo: Any = field(init=False)
    transcript: list[dict] = field(init=False)
    continuity: ContinuityRecord | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.convo = self.agent.conversation()
        self.transcript = getattr(self.convo, "turns", [])

    @classmethod
    def load(cls, agent: Any, session_id: str, path: Path) -> "Session":
        """Rebuild a session's history from its JSONL file (if present)."""
        session = cls(agent=agent, id=session_id, path=path)
        native_state_loaded = False
        native_state_transcript_len = 0
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    message = _decode_line(line)
                    if message.get("_type") == "reset":
                        session.convo = agent.conversation()
                        session.transcript = getattr(session.convo, "turns", [])
                        session.continuity = None
                        native_state_loaded = False
                    elif message.get("_type") == "runtime_turn":
                        if hasattr(agent, "conversation_from_state"):
                            session.convo = agent.conversation_from_state(
                                base64.b64decode(message["state"])
                            )
                            native_state_loaded = True
                        session.transcript.extend((message["user"], message["assistant"]))
                        native_state_transcript_len = len(session.transcript)
                    elif message.get("_type") == "approval_turn":
                        session.transcript.extend((message["user"], message["assistant"]))
                    elif message.get("_type") == "continuity":
                        session.continuity = ContinuityRecord.model_validate(message.get("record"))
                    elif message.get("role") in {"user", "assistant"}:
                        session.transcript.append(message)
        if (
            session.transcript
            and (
                not native_state_loaded
                or len(session.transcript) > native_state_transcript_len
            )
            and hasattr(agent, "conversation_from_transcript")
        ):
            session.convo = agent.conversation_from_transcript(session.transcript)
            if session.continuity is not None and hasattr(session.convo, "continuity"):
                session.convo.continuity = session.continuity
        return session

    def _append(self, *messages: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for message in messages:
                fh.write(_encode_line(message) + "\n")

    async def prepare(self, user_message: str, **kwargs) -> PendingTurn:
        """Generate a turn without making it visible to later model calls or durable history."""
        if hasattr(self.convo, "dump_state") and hasattr(self.agent, "conversation_from_state"):
            conversation = self.agent.conversation_from_state(self.convo.dump_state())
            try:
                result = await conversation.send(user_message, **kwargs)
            except Exception as exc:
                from .pydantic_runtime import PydanticRunFailure

                if not isinstance(exc, PydanticRunFailure):
                    raise
                result = exc.partial_result
            if result.status == "context_limit":
                raise ContextCapacityError("current request exceeds context capacity")
            if result.status == "approval_required":
                from .pydantic_runtime import approval_review_text

                result.text = approval_review_text(conversation.pending_approvals)
            return PendingTurn(
                user_message=user_message,
                result=result,
                runtime_state=conversation.dump_state(),
            )
        reminders = list(kwargs.pop("reminders", None) or [])
        notify_awareness = await self.agent.get_notify_awareness()
        if notify_awareness:
            reminders.append(notify_awareness)
        plan, memory_usage = await self.agent.prepare_memory_context(
            user_message, self.convo.turns, self.continuity, reminders,
        )
        if plan.continuity is not None:
            reminders.append(continuity_reminder(plan.continuity))
        result = await self.agent.run(
            user_message, history=plan.history, reminders=reminders,
            prefetched_notify_awareness=notify_awareness, **kwargs,
        )
        if result.status == "context_limit":
            raise ContextCapacityError("current request exceeds context capacity")
        merge_memory_usage(result.usage, memory_usage)
        return PendingTurn(
            user_message=user_message,
            result=result,
            continuity=plan.continuity if plan.compacted else None,
        )

    def commit(self, pending: PendingTurn) -> None:
        """Remember a prepared turn only after the channel accepted every outbound part."""
        user = {"role": "user", "content": pending.user_message, "timestamp": turn_timestamp()}
        assistant = {
            "role": "assistant",
            "content": pending.result.text,
            "citations": used_citations(
                pending.result.text, pending.result.citations,
            ),
            "timestamp": turn_timestamp(),
        }
        if pending.runtime_state is not None:
            if pending.result.status == "approval_required":
                self.transcript.extend((user, assistant))
                self._append({
                    "_type": "approval_turn",
                    "user": user,
                    "assistant": assistant,
                })
                return
            self.convo = self.agent.conversation_from_state(pending.runtime_state)
            self.transcript.extend((user, assistant))
            self._append({
                "_type": "runtime_turn",
                "state": base64.b64encode(pending.runtime_state).decode("ascii"),
                "user": user,
                "assistant": assistant,
            })
            return
        if pending.continuity is not None:
            self.continuity = pending.continuity
            self._append({
                "_type": "continuity",
                "record": pending.continuity.model_dump(),
            })
        self.transcript.extend((user, assistant))
        self._append(user, assistant)

    async def send(self, user_message: str, **kwargs) -> AgentResult:
        pending = await self.prepare(user_message, **kwargs)
        self.commit(pending)
        return pending.result

    def reset(self) -> None:
        """Start a new model-visible conversation while retaining the encrypted audit file."""
        self._append({"_type": "reset"})
        self.convo = self.agent.conversation()
        self.transcript = getattr(self.convo, "turns", [])
        self.continuity = None

    @property
    def turns(self) -> list[dict]:
        return self.transcript
