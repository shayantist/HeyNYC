"""Traces built from a CaseResult's transcript, persisted in OpenInference form.

We don't re-instrument the agent: AgentResult already retains the full message
list, so a trace is a transform of that plus the citation registry.

In-memory, `Span` keeps ergonomic fields (kind/name/input/output) so the
invariant checks read cleanly. On serialization, spans are emitted using the
**OpenInference semantic conventions**, `openinference.span.kind` plus the
canonical dotted attribute keys (`tool.name`, `tool_call.function.arguments`,
`retrieval.documents.<i>.document.content`, `llm.output_messages.<i>...`), so a
trace file can be loaded into Arize Phoenix / Langfuse without re-instrumentation.
Ref: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checks import looks_like_abstention
from .runner import CaseResult

# Tools whose output is retrieved context (RAG) rather than an action result.
RETRIEVER_TOOLS: set[str] = {"index_search"}

# Our ergonomic span kind -> the OpenInference `openinference.span.kind` value.
_SPAN_KIND = {"llm": "LLM", "tool": "TOOL", "retriever": "RETRIEVER"}

# Scope-redirect phrasing, declining because the question is out of scope.
_SCOPE_MARKERS = [
    "i help with nyc", "i help with new york", "focused on new york", "focused on nyc",
    "outside what i help", "outside of what i help", "i specialize in", "i'm here to help with",
    "i can help with nyc",
]


def classify_outcome(text: str, status: str, grounded: bool = False) -> str:
    """answered | abstained | redirected | error, what the agent ultimately did.

    HeyNYC eval metadata (not OpenInference); annotates the trace so the deterministic
    invariants can reason about the final disposition. NOTE: this keyword classifier is the
    coarse fallback, the agent-as-judge is authoritative for the semantic outcome (§A).

    A grounded, substantive answer counts as `answered` even when it *also* routes to 311 /
    the official screener: routing alongside a real, cited answer is not an abstention (this
    is the common benefits pattern, list programs, then point to access.nyc.gov)."""
    if status == "error":
        return "error"
    low = (text or "").lower()
    if grounded and len((text or "").split()) >= 40:
        return "answered"
    if any(m in low for m in _SCOPE_MARKERS):
        return "redirected"
    if looks_like_abstention(text or ""):
        return "abstained"
    return "answered"


def _as_json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


@dataclass
class Span:
    kind: str  # "llm" | "tool" | "retriever" (ergonomic; mapped to OpenInference on dump)
    name: str = ""
    input: Any = None
    output: Any = None
    model: str = ""

    def attributes(self) -> dict:
        """The OpenInference span attributes (canonical dotted keys)."""
        attrs: dict = {"openinference.span.kind": _SPAN_KIND.get(self.kind, self.kind.upper())}
        if self.kind == "tool":
            if self.name:
                attrs["tool.name"] = self.name
                attrs["tool_call.function.name"] = self.name
            if self.input is not None:
                attrs["tool_call.function.arguments"] = _as_json(self.input)
                attrs["input.value"] = _as_json(self.input)
            if self.output is not None:
                attrs["output.value"] = self.output
        elif self.kind == "retriever":
            if self.input is not None:
                attrs["input.value"] = _as_json(self.input)
            if self.output:
                # index_search returns a text blob, not structured docs → one document.
                attrs["retrieval.documents.0.document.content"] = self.output
                attrs["output.value"] = self.output
        elif self.kind == "llm":
            out = self.output or {}
            attrs["llm.output_messages.0.message.role"] = "assistant"
            if out.get("content"):
                attrs["llm.output_messages.0.message.content"] = out["content"]
            for j, tc in enumerate(out.get("tool_calls") or []):
                fn = tc.get("function", {})
                base = f"llm.output_messages.0.message.tool_calls.{j}.tool_call"
                attrs[f"{base}.id"] = tc.get("id", "")
                attrs[f"{base}.function.name"] = fn.get("name", "")
                attrs[f"{base}.function.arguments"] = fn.get("arguments", "")
            if self.model:
                attrs["llm.model_name"] = self.model
        return attrs

    def to_dict(self) -> dict:
        return {"name": self.name or self.kind, "attributes": self.attributes()}


@dataclass
class Trace:
    case_id: str
    query: str
    language: str = "en"
    spans: list[Span] = field(default_factory=list)
    final_text: str = ""
    citations: dict = field(default_factory=dict)
    outcome: str = "answered"
    turns: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "language": self.language,
            "spans": [s.to_dict() for s in self.spans],
            "final_text": self.final_text,
            "citations": self.citations,
            "outcome": self.outcome,
            "turns": self.turns,
            "diagnostics": self.diagnostics,
        }

    def write(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.case_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def build_trace(case_result: CaseResult) -> Trace:
    messages = case_result.messages or []
    tool_outputs = {
        m.get("tool_call_id"): m.get("content", "")
        for m in messages if m.get("role") == "tool"
    }
    spans: list[Span] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls") or []
        spans.append(Span(kind="llm", output={"content": m.get("content"), "tool_calls": tool_calls}))
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            call_id = tc.get("id") or name
            kind = "retriever" if name in RETRIEVER_TOOLS else "tool"
            spans.append(Span(
                kind=kind,
                name=name,
                input=_parse_args(fn.get("arguments")),
                output=tool_outputs.get(call_id, ""),
            ))
    prompts = case_result.case.turns
    turns = [
        {
            "turn": index + 1,
            "started_at": (
                case_result.turn_started_at[index]
                if index < len(case_result.turn_started_at)
                else None
            ),
            "resident_message": prompts[index] if index < len(prompts) else None,
            "text": getattr(turn, "text", ""),
            "status": getattr(turn, "status", "success"),
            "tool_calls": getattr(turn, "tool_calls_made", []),
            "citations": getattr(turn, "citations", {}),
            "messages": getattr(turn, "messages", []),
            "usage": getattr(turn, "usage", {}),
        }
        for index, turn in enumerate(case_result.turn_results)
    ]
    return Trace(
        case_id=case_result.case.id,
        query=case_result.case.query,
        language=case_result.case.language,
        spans=spans,
        final_text=case_result.text,
        citations=case_result.citations,
        outcome=classify_outcome(
            case_result.text,
            "error" if case_result.error else "success",
            grounded=bool({c.get("kind") for c in case_result.citations.values()} & {"DATA", "DOC"}),
        ),
        turns=turns,
        diagnostics=case_result.diagnostics,
    )
