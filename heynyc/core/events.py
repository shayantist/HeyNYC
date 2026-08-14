"""Agent event taxonomy for streaming a turn (token deltas + tool lifecycle).

Modeled on the convergent design across Claude Agent SDK / Codex / OpenCode:
append-style deltas for liveness (*_delta) and authoritative snapshots the client
reconciles by id (*_completed, done). Events serialize to SSE via `to_sse`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Event:
    type: ClassVar[str] = "event"

    def sse_data(self) -> dict:
        return {}

    def audit_data(self) -> dict:
        return self.sse_data()


@dataclass
class SessionInit(Event):
    type: ClassVar[str] = "session.init"
    session_id: str
    model: str

    def sse_data(self) -> dict:
        return {"session_id": self.session_id, "model": self.model}


@dataclass
class Reminder(Event):
    type: ClassVar[str] = "reminder"
    summary: str

    def sse_data(self) -> dict:
        return {"summary": self.summary}


@dataclass
class MessageStart(Event):
    type: ClassVar[str] = "message.start"
    message_id: str

    def sse_data(self) -> dict:
        return {"message_id": self.message_id}


@dataclass
class ModelRequestStart(Event):
    type: ClassVar[str] = "model.request.start"
    request_number: int

    def sse_data(self) -> dict:
        return {"request_number": self.request_number}


@dataclass
class ModelRequestCompleted(Event):
    type: ClassVar[str] = "model.request.completed"
    request_number: int
    elapsed_ms: float
    usage: dict = field(default_factory=dict)

    def sse_data(self) -> dict:
        return {
            "request_number": self.request_number,
            "elapsed_ms": self.elapsed_ms,
        }

    def audit_data(self) -> dict:
        return {**self.sse_data(), "usage": self.usage}


@dataclass
class TextDelta(Event):
    type: ClassVar[str] = "text.delta"
    message_id: str
    text: str

    def sse_data(self) -> dict:
        return {"message_id": self.message_id, "text": self.text}


@dataclass
class ToolStart(Event):
    type: ClassVar[str] = "tool.start"
    tool_call_id: str
    name: str
    label: str = ""
    args: dict = field(default_factory=dict)

    def sse_data(self) -> dict:
        return {"tool_call_id": self.tool_call_id, "name": self.name, "label": self.label}

    def audit_data(self) -> dict:
        return {**self.sse_data(), "args": self.args}


@dataclass
class ToolCompleted(Event):
    type: ClassVar[str] = "tool.completed"
    tool_call_id: str
    name: str
    status: str  # "ok" | "error"
    result_summary: str = ""
    result: Any = None

    def sse_data(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "result_summary": self.result_summary,
        }

    def audit_data(self) -> dict:
        return {**self.sse_data(), "result": self.result}


@dataclass
class OutputAttempt(Event):
    type: ClassVar[str] = "output.attempt"
    tool_call_id: str
    name: str
    args: dict = field(default_factory=dict)

    def sse_data(self) -> dict:
        return {"tool_call_id": self.tool_call_id, "name": self.name}

    def audit_data(self) -> dict:
        return {**self.sse_data(), "args": self.args}


@dataclass
class ValidationRejected(Event):
    type: ClassVar[str] = "validation.rejected"
    tool_call_id: str
    name: str
    message: str

    def sse_data(self) -> dict:
        return {"tool_call_id": self.tool_call_id, "name": self.name}

    def audit_data(self) -> dict:
        return {**self.sse_data(), "message": self.message}


@dataclass
class ToolApprovalRequired(Event):
    type: ClassVar[str] = "tool.approval_required"
    tool_call_id: str
    name: str
    args: dict

    def sse_data(self) -> dict:
        return {"tool_call_id": self.tool_call_id, "name": self.name, "args": self.args}


@dataclass
class MessageCompleted(Event):
    type: ClassVar[str] = "message.completed"
    message_id: str
    text: str
    citations: dict = field(default_factory=dict)

    def sse_data(self) -> dict:
        return {"message_id": self.message_id, "text": self.text, "citations": self.citations}


@dataclass
class ErrorEvent(Event):
    type: ClassVar[str] = "error"
    scope: str  # "model" | "tool"
    message: str
    retryable: bool = False

    def sse_data(self) -> dict:
        return {"scope": self.scope, "message": self.message, "retryable": self.retryable}


@dataclass
class TurnAborted(Event):
    type: ClassVar[str] = "turn.aborted"
    reason: str = "user_interrupt"

    def sse_data(self) -> dict:
        return {"reason": self.reason}


@dataclass
class Done(Event):
    type: ClassVar[str] = "done"
    status: str  # "success" | "max_turns" | "max_budget" | "error" | "aborted"
    num_turns: int
    citations: dict = field(default_factory=dict)
    result: Any = None  # the AgentResult (not serialized over SSE)

    def sse_data(self) -> dict:
        return {"status": self.status, "num_turns": self.num_turns, "citations": self.citations}


def to_sse(event: Event) -> str:
    """Render an event as an SSE frame."""
    import json

    return f"event: {event.type}\ndata: {json.dumps(event.sse_data())}\n\n"
