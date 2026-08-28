"""Typed tool contract shared by all modules.

A Tool wraps an async handler with a Pydantic input model. It serializes to two
standard, open formats so HeyNYC tools are portable across agent frameworks:

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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, TypeAdapter

from ..citations import CitationRegistry
from ..registry import Registry

if TYPE_CHECKING:
    from .geo import GeoPoint


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
    current_location: Optional["GeoPoint"] = None
    event_retrieval_policy: Literal["fast", "deep"] = "fast"
    toolbox: Optional[dict[str, Any]] = None  # existing sibling tools for bounded module coordinators
    http: Optional[Any] = None  # httpx.AsyncClient; None → tools create their own
    embedder: Optional[Any] = None  # index Embedder; tools that retrieve reuse the production default
    retrieval_cache_path: Optional[Any] = None  # persistent Lance cache for live structured catalogs
    evidence_token_budget: int | None = None  # model-specific input capacity left for one retrieval
    evidence_tokens_used: int = 0  # ranked retrieval evidence already added in this resident turn
    evidence_model: str | None = None
    output_dir: Optional[Any] = None  # tools that emit a file (e.g. a filled PDF) write here; the channel sends it
    drafts: Optional[Any] = None  # per-user structured draft accessor (UserDrafts); persists in-progress form slots
    event_turn: Optional[str] = None  # semantic scope-preflight tri-state: none|discovery|preparation (None → tool falls back to its regexes)
    current_turn_modules: frozenset[str] = frozenset()
    current_turn_capability_ids: frozenset[str] = frozenset()
    current_turn_high_stakes: bool = False
    allow_unverified_search_excerpts: bool = False
    delivered_notify_titles: frozenset = frozenset()  # F080: normalized Notify titles already cited earlier in THIS conversation; a repeat advisories call answers with a marker, not a re-brief
    resident_facts: dict[str, ResidentFact] = field(default_factory=dict)
    fact_review_runs: list[dict[str, Any]] = field(default_factory=list)
    claim_support_runs: list[dict[str, Any]] = field(default_factory=list)
    tool_runs: list[dict[str, Any]] = field(default_factory=list)
    # Turn-local only: exact duplicate reads share one snapshot, never cross-turn freshness.
    tool_result_cache: dict[tuple[str, str], Any] = field(default_factory=dict)
    tool_result_tasks: dict[tuple[str, str], Any] = field(default_factory=dict)
    language_verifier_runs: list[dict[str, Any]] = field(default_factory=list)
    validation_rejections: list[dict[str, Any]] = field(default_factory=list)
    tool_result_urls: set[str] = field(default_factory=set)
    tool_result_citation_ids: set[str] = field(default_factory=set)
    response_priority_citation_ids: set[str] = field(default_factory=set)
    required_response_citation_ids: set[str] = field(default_factory=set)
    rendered_fetch_urls: set[str] = field(default_factory=set)
    language: str | None = None
    verify_output_language: bool = False
    cooling_terminal_result: str | None = None
    cooling_terminal_citation_ids: tuple[str, ...] = ()
    cooling_terminal_synthesis: bool = False


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[Any]]


class ToolInput(BaseModel):
    """Model-facing tool input. Unknown arguments are always errors."""

    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return self.model_dump(mode="python", exclude_none=True).items()

    def keys(self):
        return self.model_dump(mode="python", exclude_none=True).keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


class EmptyToolInput(ToolInput):
    pass


class ToolFailure(BaseModel):
    """Expected operational failure returned to the agent."""

    status: Literal["unavailable", "partial", "rejected"]
    reason: str
    retryable: bool
    source_url: AnyHttpUrl | None = None


class ToolFailureError(RuntimeError):
    """Expected failure raised inside nested tool helpers."""

    def __init__(
        self,
        *,
        status: Literal["unavailable", "partial", "rejected"],
        reason: str,
        retryable: bool,
        source_url: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.failure = ToolFailure(
            status=status,
            reason=reason,
            retryable=retryable,
            source_url=source_url,
        )


@dataclass
class Tool:
    name: str
    description: str
    handler: ToolHandler
    input_type: type[BaseModel] = EmptyToolInput
    # MCP annotation hints (https://modelcontextprotocol.io/...): behavioral
    # metadata clients use to decide auto-approval, parallelism, caching, etc.
    read_only: bool = True       # readOnlyHint , doesn't modify state
    destructive: bool = False    # destructiveHint, destroys/overwrites data
    idempotent: bool = True      # idempotentHint, safe to retry with same args
    open_world: bool = False     # openWorldHint , hits external/open-ended data
    requires_approval: bool = False  # gate side-effecting tools behind user approval
    strict: bool | None = None   # let the provider enforce strictness when the schema supports it
    title: str = ""              # human-readable label
    module: str = ""             # owning service module; "" = core, always exposed (diet block 4)
    resident_fact_scope: tuple[str, ...] = ()  # JSON-pointer roots that must match trusted resident facts
    return_type: Any = None       # declared Pydantic-serializable result; None keeps legacy strings
    result_handler: ToolHandler | None = None  # internal normalized results before model rendering

    def __post_init__(self) -> None:
        if (not self.read_only or self.destructive) and not self.requires_approval:
            raise ValueError("Side-effecting tools must set requires_approval=True")

    def _input_schema(self) -> dict:
        """Parameter schema with `additionalProperties: false` (strict-schema best practice)."""
        params = TypeAdapter(self.input_type).json_schema()
        params.setdefault("additionalProperties", False)
        return params

    async def invoke(self, raw_args: object, ctx: ToolContext) -> Any:
        request = TypeAdapter(self.input_type).validate_python(raw_args, extra="forbid")
        try:
            return await self.handler(request, ctx)
        except ToolFailureError as exc:
            return exc.failure

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
