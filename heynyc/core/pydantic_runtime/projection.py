from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.usage import RunUsage

from heynyc.core.telemetry import priced_cost_usd

_GROUNDED_OUTPUT_TOOL = "grounded_answer"
_NONFACTUAL_OUTPUT_TOOL = "nonfactual_outcome"
_CLARIFICATION_OUTPUT_TOOL = "clarification_request"
NONFACTUAL_OUTCOME_TEXT = (
    "I can't know that yet. I can help with the practical NYC part instead."
)
_OUTPUT_TOOLS = {
    _GROUNDED_OUTPUT_TOOL,
    _NONFACTUAL_OUTPUT_TOOL,
    _CLARIFICATION_OUTPUT_TOOL,
}
_SEMANTIC_CITATION_CHARS = 1_200
_LEGACY_CITATION_RE = re.compile(
    r"\{\s*cite\s*:\s*(S\d+)\s*\}",
    re.IGNORECASE,
)

class GroundedBlock(BaseModel):
    text: str = Field(
        min_length=1,
        description="One factual or procedural claim, without citation markers.",
    )
    citation_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description="IDs of retrieved sources that support the whole claim.",
    )


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grounded_blocks: list[GroundedBlock] = Field(
        min_length=1,
        max_length=12,
    )


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)


def _grounded_block_text(block: GroundedBlock) -> str:
    declared = {citation_id.casefold() for citation_id in block.citation_ids}
    return _LEGACY_CITATION_RE.sub(
        lambda match: "" if match.group(1).casefold() in declared else match.group(),
        block.text,
    ).strip()


def _legacy_citation_ids(text: str) -> list[str]:
    return [match.group(1).upper() for match in _LEGACY_CITATION_RE.finditer(text)]


def _semantic_citation_evidence(citation: dict) -> str:
    """Prefer the tool's bounded evidence chunk; keep full snapshots for deterministic audit."""
    return " ".join(
        str(citation.get(field) or "").strip()
        for field in ("snippet", "title")
    ).strip()[:_SEMANTIC_CITATION_CHARS]


def _render_grounded_answer(answer: GroundedAnswer) -> str:
    return "\n\n".join(
        " ".join(
            (
                _grounded_block_text(block),
                " ".join(
                    f"{{cite:{citation_id}}}"
                    for citation_id in dict.fromkeys(block.citation_ids)
                ),
            )
        )
        for block in answer.grounded_blocks
    )


def _accepted_grounded_outputs(
    messages: Sequence[ModelMessage],
) -> dict[str, str]:
    accepted = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name in _OUTPUT_TOOLS
    }
    return {
        part.tool_call_id: (
            _render_grounded_answer(
                GroundedAnswer.model_validate(part.args_as_dict())
            )
            if part.tool_name == _GROUNDED_OUTPUT_TOOL
            else (
                ClarificationRequest.model_validate(part.args_as_dict()).question
                if part.tool_name == _CLARIFICATION_OUTPUT_TOOL
                else NONFACTUAL_OUTCOME_TEXT
            )
        )
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
        and part.tool_name in _OUTPUT_TOOLS
        and part.tool_call_id in accepted
    }


def _response_text(
    message: ModelResponse,
    accepted_outputs: dict[str, str],
) -> str:
    structured = [
        accepted_outputs[part.tool_call_id]
        for part in message.parts
        if isinstance(part, ToolCallPart)
        and part.tool_name in _OUTPUT_TOOLS
        and part.tool_call_id in accepted_outputs
    ]
    return "".join(structured or [
        part.content for part in message.parts if isinstance(part, TextPart)
    ])


def _openai_messages(
    messages: Sequence[ModelMessage],
    *,
    accepted_outputs: dict[str, str] | None = None,
) -> list[dict]:
    """Translate native PydanticAI messages into HeyNYC's existing trace contract."""
    accepted_outputs = (
        _accepted_grounded_outputs(messages)
        if accepted_outputs is None
        else accepted_outputs
    )
    translated: list[dict] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = _response_text(message, accepted_outputs)
            calls = [
                {
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": part.tool_name,
                        "arguments": (
                            part.args
                            if isinstance(part.args, str)
                            else json.dumps(part.args, separators=(",", ":"))
                        ),
                    },
                }
                for part in message.parts
                if isinstance(part, (ToolCallPart, NativeToolSearchCallPart))
                and part.tool_name not in _OUTPUT_TOOLS
            ]
            translated.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": calls or None,
                }
            )
            translated.extend(
                {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": json.dumps(part.content, separators=(",", ":")),
                }
                for part in message.parts
                if isinstance(part, NativeToolSearchReturnPart)
            )
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    translated.append({"role": "user", "content": part.content})
                elif isinstance(
                    part,
                    (
                        ToolReturnPart,
                        RetryPromptPart,
                        NativeToolSearchReturnPart,
                    ),
                ) and part.tool_name not in _OUTPUT_TOOLS:
                    translated.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": str(part.content),
                        }
                    )
    return translated


def _measurement_messages(
    messages: Sequence[ModelMessage],
    *,
    omit_instruction: str = "",
) -> list[dict]:
    """Include request controls that the public trace intentionally omits."""
    accepted_outputs = _accepted_grounded_outputs(messages)
    translated: list[dict] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            translated.extend(
                {"role": "system", "content": part.content}
                for part in message.parts
                if isinstance(part, SystemPromptPart)
                and part.content
            )
            if message.instructions:
                instructions = message.instructions
                if omit_instruction:
                    instructions = instructions.replace(
                        omit_instruction,
                        "",
                        1,
                    ).strip()
                if instructions:
                    translated.append(
                        {"role": "system", "content": instructions}
                    )
        translated.extend(
            _openai_messages([message], accepted_outputs=accepted_outputs)
        )
    return translated


def _native_cost(messages: Sequence[ModelMessage]) -> float | None:
    """Sum PydanticAI's native per-response prices, or mark the run unpriced."""
    total = 0.0
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        try:
            total += float(message.cost().total_price)
        except (AssertionError, LookupError, ValueError):
            return None
    return total


def _complete_cost(
    model: str,
    messages: Sequence[ModelMessage],
    usage: Any,
) -> tuple[float | None, str]:
    current = priced_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
    )
    if current is not None:
        return current, "litellm-fallback"
    native = _native_cost(messages)
    if native is not None:
        return native, "pydantic-native"
    return None, "unpriced"


def _captured_usage(messages: Sequence[ModelMessage]) -> RunUsage:
    usage = RunUsage()
    for message in messages:
        if isinstance(message, ModelResponse):
            usage.incr(message.usage)
            usage.requests += 1
    usage.tool_calls = sum(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in message.parts
    )
    return usage


def _retry_kinds(messages: Sequence[ModelMessage]) -> list[str]:
    prefixes = {
        "Answer with at least one grounded block": "missing_grounded_blocks",
        "Use only citation IDs returned": "unknown_citation",
        "When a grounded block includes legacy citation markers": "citation_mismatch",
        "Do not write citation markers": "citation_marker",
        "Search snippets are discovery only": "discovery_only",
        "Return a complete replacement answer to the resident's full request": (
            "deterministic_grounding"
        ),
        "Return a complete replacement answer. Keep every supported outcome": "semantic_grounding",
        "The resident wrote primarily": "reply_script",
    }
    return [
        next(
            (
                kind
                for prefix, kind in prefixes.items()
                if str(part.content).startswith(prefix)
            ),
            "output_validation",
        )
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]


def _native_cache_settings(model: Any) -> dict[str, Any]:
    """Bound requests and add provider-native prompt cache controls."""
    settings: dict[str, Any] = {"timeout": 60}
    system = getattr(model, "system", "")
    if system == "openai":
        settings["openai_prompt_cache_key"] = "heynyc-pydantic-v1"
    elif system == "anthropic":
        settings["anthropic_cache_instructions"] = True
    return settings


def _dynamic_instructions(parts: Sequence[str]) -> Callable[[], str] | None:
    """Keep per-turn context after provider prompt-cache boundaries."""
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return (lambda: text) if text else None


def _resident_history(messages: Sequence[ModelMessage]) -> list[dict]:
    """Collapse native traces to complete resident/final-assistant exchanges."""
    accepted_outputs = _accepted_grounded_outputs(messages)
    history: list[dict] = []
    user: str | None = None
    assistant: str | None = None
    for message in messages:
        prompts = [
            part.content
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        if prompts:
            if user is not None and assistant is not None:
                history.extend(
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                )
            user, assistant = prompts[-1], None
        if isinstance(message, ModelResponse):
            text = _response_text(message, accepted_outputs)
            if text:
                assistant = text
    if user is not None and assistant is not None:
        history.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    return history


def _native_history(history: Sequence[dict]) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(message["content"])])
        if message["role"] == "user"
        else ModelResponse(parts=[TextPart(message["content"])])
        for message in history
    ]


def _native_orchestration_history(
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    """Keep PydanticAI state that its capability and tool-search loaders replay."""
    preserved: list[ModelMessage] = []
    response_parts = (
        LoadCapabilityCallPart,
        ToolSearchCallPart,
        NativeToolSearchCallPart,
        NativeToolSearchReturnPart,
    )
    request_parts = (LoadCapabilityReturnPart, ToolSearchReturnPart)
    for message in messages:
        if isinstance(message, ModelResponse):
            has_state = any(
                isinstance(part, response_parts) for part in message.parts
            )
            parts = [
                part
                for part in message.parts
                if has_state and isinstance(part, (ThinkingPart, *response_parts))
            ]
        else:
            parts = [
                part for part in message.parts if isinstance(part, request_parts)
            ]
        if parts:
            preserved.append(replace(message, parts=parts))
    return preserved


def _function_tool_schemas(request_context: ModelRequestContext) -> list[dict]:
    """Translate only function tools exposed on this exact native request."""
    schemas = []
    for tool in request_context.model_request_parameters.function_tools:
        if tool.defer_loading:
            continue
        function = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters_json_schema,
        }
        if tool.strict:
            function["strict"] = True
        schemas.append({"type": "function", "function": function})
    return schemas
