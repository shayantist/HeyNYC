"""Tool abstraction shared by all modules.

A Tool wraps an async handler with a JSON-Schema parameter spec. It serializes to
two standard, open formats so HeyNYC tools are portable across agent frameworks:

- `schema()` → the OpenAI / Anthropic / litellm function-calling shape (what the
  agent passes to the model). We inject `additionalProperties: false` so the model
  can't improvise arguments (OpenAI/Anthropic strict-schema best practice).
- `to_mcp()` → the Model Context Protocol tool shape (`name` / `description` /
  `inputSchema` / `annotations`), so the same toolbox can be exposed via an MCP
  server to any MCP client (Claude, Cursor, OpenAI, …). MCP is the open
  cross-framework standard; our Tool maps to it 1:1.

Handlers receive parsed args and a ToolContext giving access to the citation
registry, the module registry, and (in tests) an injected HTTP client so network
calls can be mocked.

Refs: https://modelcontextprotocol.io/specification/2025-06-18/schema ;
https://www.anthropic.com/engineering/writing-tools-for-agents
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from ..citations import CitationRegistry
from ..registry import Registry


@dataclass(frozen=True)
class ResidentFact:
    """A typed resident value captured by application code, never by the model."""

    value: Any
    source_turn_id: str
    status: Literal["captured", "confirmed"]


@dataclass
class ToolContext:
    citations: CitationRegistry
    registry: Registry
    query: str = ""  # current resident turn; lets tools resolve relative constraints deterministically
    user_history: str = ""  # resident-authored turns only; validates model-supplied tool arguments
    user_turns: tuple[str, ...] = ()  # structured resident turns; avoids stale-location substring matches
    toolbox: Optional[dict[str, Any]] = None  # existing sibling tools for bounded module coordinators
    http: Optional[Any] = None  # httpx.AsyncClient; None → tools create their own
    embedder: Optional[Any] = None  # index Embedder; tools that retrieve reuse it (None → default/Hash)
    output_dir: Optional[Any] = None  # tools that emit a file (e.g. a filled PDF) write here; the channel sends it
    drafts: Optional[Any] = None  # per-user structured draft accessor (UserDrafts); persists in-progress form slots
    event_turn: Optional[str] = None  # semantic scope-preflight tri-state: none|discovery|preparation (None → tool falls back to its regexes)
    delivered_notify_titles: frozenset = frozenset()  # F080: normalized Notify titles already cited earlier in THIS conversation; a repeat advisories call answers with a marker, not a re-brief
    resident_facts: dict[str, ResidentFact] = field(default_factory=dict)
    fact_review_runs: list[dict[str, Any]] = field(default_factory=list)
    semantic_verifier_runs: list[dict[str, Any]] = field(default_factory=list)
    validation_rejections: list[dict[str, Any]] = field(default_factory=list)
    response_priority_citation_ids: set[str] = field(default_factory=set)
    language: str | None = None
    cooling_terminal_result: str | None = None
    cooling_terminal_citation_ids: tuple[str, ...] = ()


ToolHandler = Callable[[dict, ToolContext], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object
    handler: ToolHandler
    # MCP annotation hints (https://modelcontextprotocol.io/...): behavioral
    # metadata clients use to decide auto-approval, parallelism, caching, etc.
    read_only: bool = True       # readOnlyHint , doesn't modify state
    destructive: bool = False    # destructiveHint, destroys/overwrites data
    idempotent: bool = True      # idempotentHint, safe to retry with same args
    open_world: bool = False     # openWorldHint , hits external/open-ended data
    requires_approval: bool = False  # gate side-effecting tools behind user approval
    strict: bool = False         # emit `strict: true` (all params required) for constrained decoding
    title: str = ""              # human-readable label
    module: str = ""             # owning service module; "" = core, always exposed (diet block 4)
    resident_fact_scope: tuple[str, ...] = ()  # JSON-pointer roots that must match trusted resident facts

    def __post_init__(self) -> None:
        if (not self.read_only or self.destructive) and not self.requires_approval:
            raise ValueError("Side-effecting tools must set requires_approval=True")

    def _input_schema(self) -> dict:
        """Parameter schema with `additionalProperties: false` (strict-schema best practice)."""
        params = dict(self.parameters)
        params.setdefault("additionalProperties", False)
        return params

    def schema(self) -> dict:
        """OpenAI / Anthropic / litellm function-calling schema."""
        fn = {"name": self.name, "description": self.description, "parameters": self._input_schema()}
        if self.strict:
            fn["strict"] = True
        return {"type": "function", "function": fn}

    def to_mcp(self) -> dict:
        """Model Context Protocol tool object, portable to any MCP client."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self._input_schema(),
            "annotations": {
                "title": self.title or self.name,
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.idempotent,
                "openWorldHint": self.open_world,
            },
        }
