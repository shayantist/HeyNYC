"""Compact eval summaries built from the resident-facing result projection.

These summaries support deterministic eval invariants. They are not the full agent
loop. Live evals also export Pydantic AI's native OpenTelemetry spans, including
model inputs, outputs, tool definitions, calls, results, and agent message history.

In-memory, `Span` keeps ergonomic fields (kind/name/input/output) so the
invariant checks read cleanly. On serialization, spans are emitted using the
**OpenInference semantic conventions**, `openinference.span.kind` plus the
canonical dotted attribute keys (`tool.name`, `tool_call.function.arguments`,
`retrieval.documents.<i>.document.content`, `llm.output_messages.<i>...`), so a
summary file can be loaded into Arize Phoenix / Langfuse.
Ref: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from opentelemetry.trace import TracerProvider
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_core import to_jsonable_python

from heynyc.core.pii_redaction import redact_sensitive_identifiers

from .runner import CaseResult

# Tools whose output is retrieved context (RAG) rather than an action result.
RETRIEVER_TOOLS: set[str] = {"index_search"}

# Our ergonomic span kind -> the OpenInference `openinference.span.kind` value.
_SPAN_KIND = {"llm": "LLM", "tool": "TOOL", "retriever": "RETRIEVER"}


def eval_otel_exporter(
    path: Path,
) -> tuple[InstrumentationSettings, TracerProvider, IO[str]]:
    """Export Pydantic AI's native, content-bearing OTel spans as JSON lines."""
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w", encoding="utf-8")
    provider = SdkTracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(
            ConsoleSpanExporter(
                out=stream,
                formatter=lambda span: span.to_json(indent=None) + "\n",
            )
        )
    )
    return (
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=True,
            include_binary_content=False,
        ),
        provider,
        stream,
    )

def classify_outcome(text: str, status: str, grounded: bool = False) -> str:
    """Record only mechanically known completion status.

    Whether a completed response answered, abstained, or redirected is semantic and belongs in
    the full-trace qualitative review.
    """
    del text, grounded
    if status == "error":
        return "error"
    return "unclassified"


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
    redteam_category: str = ""
    adversarial_intent: str = ""
    safety_criterion: str = ""
    spans: list[Span] = field(default_factory=list)
    final_text: str = ""
    citations: dict = field(default_factory=dict)
    outcome: str = "unclassified"
    turns: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "language": self.language,
            "redteam_category": self.redteam_category,
            "adversarial_intent": self.adversarial_intent,
            "safety_criterion": self.safety_criterion,
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
        path.write_text(json.dumps(to_jsonable_python(self.to_dict()), indent=2))
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
            "resident_message": (
                redact_sensitive_identifiers(prompts[index])
                if index < len(prompts)
                else None
            ),
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
        query=redact_sensitive_identifiers(case_result.case.query),
        language=case_result.case.language,
        redteam_category=case_result.case.redteam_category,
        adversarial_intent=case_result.case.adversarial_intent,
        safety_criterion=case_result.case.safety_criterion,
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
