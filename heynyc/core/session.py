"""Session persistence — save/resume conversations as append-only JSONL.

Append-only JSONL (one message per line) is the proven, simple format used by
Claude Code et al.: cheap to append, easy to replay/rebuild on resume. A Session
wraps a Conversation and persists each committed turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent import Agent, AgentResult, Conversation


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
                    session.convo.turns.append(json.loads(line))
        return session

    def _append(self, *messages: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for message in messages:
                fh.write(json.dumps(message) + "\n")

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
