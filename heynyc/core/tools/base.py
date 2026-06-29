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

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..citations import CitationRegistry
from ..registry import Registry


@dataclass
class ToolContext:
    citations: CitationRegistry
    registry: Registry
    http: Optional[Any] = None  # httpx.AsyncClient; None → tools create their own
    embedder: Optional[Any] = None  # index Embedder; tools that retrieve reuse it (None → default/Hash)


ToolHandler = Callable[[dict, ToolContext], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object
    handler: ToolHandler
    # MCP annotation hints (https://modelcontextprotocol.io/...): behavioral
    # metadata clients use to decide auto-approval, parallelism, caching, etc.
    read_only: bool = True       # readOnlyHint  — doesn't modify state
    destructive: bool = False    # destructiveHint — destroys/overwrites data
    idempotent: bool = True      # idempotentHint — safe to retry with same args
    open_world: bool = False     # openWorldHint  — hits external/open-ended data
    requires_approval: bool = False  # gate side-effecting tools behind user approval
    strict: bool = False         # emit `strict: true` (all params required) for constrained decoding
    title: str = ""              # human-readable label

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
        """Model Context Protocol tool object — portable to any MCP client."""
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
