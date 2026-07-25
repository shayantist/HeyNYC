from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, Awaitable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator
from litellm.main import responses_api_bridge_check
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    AgentStreamEvent,
    DeferredToolRequests,
    DeferredToolRequestsEvent,
    DeferredToolResults,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRetry,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ToolOutput,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UsageLimits,
    capture_run_messages,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    ReinjectSystemPrompt,
    WrapModelRequestHandler,
)
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
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
from pydantic_ai.models import ModelRequestContext, infer_model
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.tools import Tool as PydanticTool
from pydantic_ai.usage import RunUsage

from heynyc.core import config, events
from heynyc.core.agent import (
    AgentResult,
    _emergency_backstop,
    _internal_config_backstop,
    _reply_script_feedback,
    _sensitive_identifier_backstop,
)
from heynyc.core.citations import CitationRegistry, used_citations
from heynyc.core.freshness import attach_temporal_provenance
from heynyc.core.grounding import check_grounding
from heynyc.core.memory import (
    CompactFn,
    ContextCapacityError,
    ContinuityRecord,
    MeasureFn,
    compact_memory,
    context_capacity,
    continuity_reminder,
    merge_memory_usage,
    prepare_context,
    request_tokens,
)
from heynyc.core.nli import NLIInput
from heynyc.core.prompts import build_system_prompt_tiers
from heynyc.core.registry import Registry
from heynyc.core.spend import SpendGuard
from heynyc.core.telemetry import priced_cost_usd
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.base import ResidentFact, Tool, ToolContext

_DEFERRED_REQUESTS = TypeAdapter(DeferredToolRequests)
_RESIDENT_FACTS = TypeAdapter(dict[str, ResidentFact])
_GROUNDED_OUTPUT_TOOL = "grounded_answer"
_SEMANTIC_EVIDENCE_CHARS = 1_200
_STRUCTURED_GROUNDING_SYSTEM_PROMPT = (
    "For the final GroundedAnswer output, do not write inline citation markers. "
    "Put retrieved source IDs only in citation_ids. The runtime renders citation "
    "markers after validation."
)
TEMPORARY_FAILURE_FALLBACK = (
    "I hit a temporary problem before I could verify an answer. "
    "Please try again in a moment."
)


def _emit(
    sink: Callable[[events.Event], None] | None,
    event: events.Event,
) -> None:
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        pass


def _finish_events(
    sink: Callable[[events.Event], None] | None,
    message_id: str,
    result: AgentResult,
) -> None:
    if result.status == "error":
        _emit(
            sink,
            events.ErrorEvent(
                scope="model",
                message="The model run ended before a verified answer was ready.",
                retryable=True,
            ),
        )
    _emit(
        sink,
        events.MessageCompleted(
            message_id=message_id,
            text=result.text,
            citations=result.citations,
        ),
    )
    _emit(
        sink,
        events.Done(
            status=result.status,
            num_turns=result.iterations,
            citations=result.citations,
            result=result,
        ),
    )


async def _forward_events(
    sink: Callable[[events.Event], None],
    message_id: str,
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        if isinstance(event, FunctionToolCallEvent):
            _emit(
                sink,
                events.ToolStart(
                    tool_call_id=event.part.tool_call_id,
                    name=event.part.tool_name,
                ),
            )
        elif isinstance(event, FunctionToolResultEvent):
            part = event.part
            _emit(
                sink,
                events.ToolCompleted(
                    tool_call_id=part.tool_call_id,
                    name=part.tool_name,
                    status="ok" if isinstance(part, ToolReturnPart) else "error",
                    result_summary=str(event.content or "")[:160],
                ),
            )
        elif isinstance(event, DeferredToolRequestsEvent):
            for call in event.requests.approvals:
                _emit(
                    sink,
                    events.ToolApprovalRequired(
                        tool_call_id=call.tool_call_id,
                        name=call.tool_name,
                        args=call.args_as_dict(),
                    ),
                )
        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            if event.part.content:
                _emit(
                    sink,
                    events.TextDelta(message_id=message_id, text=event.part.content),
                )
        elif isinstance(event, PartDeltaEvent) and isinstance(
            event.delta, TextPartDelta
        ):
            if event.delta.content_delta:
                _emit(
                    sink,
                    events.TextDelta(
                        message_id=message_id,
                        text=event.delta.content_delta,
                    ),
                )


class GroundedBlock(BaseModel):
    text: str = Field(
        min_length=1,
        description="One factual or procedural claim, without citation markers.",
    )
    citation_ids: list[str] = Field(
        min_length=1,
        description="IDs of retrieved sources that support the whole claim.",
    )


class GroundedAnswer(BaseModel):
    acknowledgment: str = Field(
        default="",
        max_length=240,
        description=(
            "A brief empathetic reaction and, when needed, an explicit limitation on "
            "what can be determined, in the resident's language. No external factual "
            "or procedural claims, advice, predictions, eligibility, actions, names, "
            "or dates."
        ),
    )
    grounded_blocks: list[GroundedBlock] = Field(default_factory=list)
    follow_up_question: str = Field(
        default="",
        max_length=240,
        description=(
            "An optional neutral clarification question. It may include a narrow "
            "data-minimization reminder not to share sensitive identifiers. Do not "
            "add other factual or procedural claims or direct another action."
        ),
    )


class PydanticRunFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        partial_result: AgentResult,
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result
        self.diagnostics = diagnostics


def _grounded_block_text(block: GroundedBlock) -> str:
    text = block.text
    for citation_id in dict.fromkeys(block.citation_ids):
        text = text.replace(f"{{cite:{citation_id}}}", "")
    return text.strip()


def _semantic_citation_evidence(citation: dict) -> str:
    """Prefer the tool's bounded evidence chunk; keep full snapshots for deterministic audit."""
    return " ".join(
        str(citation.get(field) or "").strip()
        for field in ("snippet", "title")
    ).strip()[:_SEMANTIC_EVIDENCE_CHARS]


def _render_grounded_answer(answer: GroundedAnswer) -> str:
    parts = [answer.acknowledgment.strip()]
    parts.extend(
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
    parts.append(answer.follow_up_question.strip())
    return "\n\n".join(part for part in parts if part)


def _accepted_grounded_outputs(
    messages: Sequence[ModelMessage],
) -> dict[str, str]:
    accepted = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name == _GROUNDED_OUTPUT_TOOL
    }
    return {
        part.tool_call_id: _render_grounded_answer(
            GroundedAnswer.model_validate(part.args_as_dict())
        )
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
        and part.tool_name == _GROUNDED_OUTPUT_TOOL
        and part.tool_call_id in accepted
    }


def _response_text(
    message: ModelResponse,
    accepted_outputs: dict[str, str],
) -> str:
    parts = [part.content for part in message.parts if isinstance(part, TextPart)]
    for part in message.parts:
        if (
            isinstance(part, ToolCallPart)
            and part.tool_name == _GROUNDED_OUTPUT_TOOL
            and part.tool_call_id in accepted_outputs
        ):
            parts.append(accepted_outputs[part.tool_call_id])
    return "".join(parts)


def _fact_leaves(value: object, path: str) -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _fact_leaves(
                child,
                f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}",
            )
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _fact_leaves(child, f"{path}/{index}")
        ]
    return [(path, value)]


def _resident_fact_errors(
    args: dict[str, object],
    ctx: ToolContext,
    scopes: Sequence[str],
) -> list[str]:
    leaves = [
        leaf
        for scope in scopes
        if scope.removeprefix("/") in args
        for leaf in _fact_leaves(args[scope.removeprefix("/")], scope)
    ]
    return [
        path
        for path, value in leaves
        if (fact := ctx.resident_facts.get(path)) is None
        or type(fact.value) is not type(value)
        or fact.value != value
    ]


def adapt_tool(tool: Tool) -> PydanticTool:
    """Wrap one existing HeyNYC tool without changing its handler or schema."""
    schema = tool._input_schema()
    validator = Draft202012Validator(schema)

    def validate(ctx: RunContext[ToolContext], **kwargs: object) -> None:
        errors = sorted(
            validator.iter_errors(kwargs), key=lambda error: list(error.path)
        )
        if errors:
            raise ModelRetry(f"Invalid arguments for {tool.name}: {errors[0].message}")
        unsupported = _resident_fact_errors(
            kwargs,
            ctx.deps,
            tool.resident_fact_scope,
        )
        if unsupported:
            paths = ", ".join(unsupported)
            raise ModelRetry(
                f"Resident evidence is missing or differs for {paths}. "
                f"Omit unknown optional fields or call confirm_{tool.name}_facts "
                "with the exact profile for resident confirmation."
            )

    async def invoke(ctx: RunContext[ToolContext], **kwargs: object) -> str:
        return await tool.handler(dict(kwargs), ctx.deps)

    adapted = PydanticTool.from_schema(
        invoke,
        name=tool.name,
        description=tool.description,
        json_schema=schema,
        takes_ctx=True,
        args_validator=validate,
    )
    adapted.requires_approval = tool.requires_approval
    adapted.strict = tool.strict
    if tool.resident_fact_scope:

        def expose_after_confirmation(ctx: RunContext[ToolContext], definition: Any):
            has_scoped_facts = any(
                any(
                    path == scope or path.startswith(f"{scope}/")
                    for scope in tool.resident_fact_scope
                )
                for path in ctx.deps.resident_facts
            )
            return definition if has_scoped_facts else None

        adapted.prepare = expose_after_confirmation
    adapted.metadata = {
        "title": tool.title or tool.name,
        "readOnlyHint": tool.read_only,
        "destructiveHint": tool.destructive,
        "idempotentHint": tool.idempotent,
        "openWorldHint": tool.open_world,
        "heynyc_module": tool.module,
    }
    return adapted


def resident_fact_confirmation_tool(tool: Tool) -> Tool:
    """Reuse a governed tool's schema for native structured fact confirmation."""
    if not tool.read_only or tool.destructive or not tool.idempotent:
        raise ValueError(
            "Resident fact confirmation can only wrap read-only idempotent tools"
        )

    async def confirm(args: dict, ctx: ToolContext) -> str:
        source_turn_id = f"turn-{len(ctx.user_turns)}"
        for scope in tool.resident_fact_scope:
            key = scope.removeprefix("/")
            if key not in args:
                continue
            for path, value in _fact_leaves(args[key], scope):
                ctx.resident_facts[path] = ResidentFact(
                    value=value,
                    source_turn_id=source_turn_id,
                    status="confirmed",
                )
        return await tool.handler(args, ctx)

    return Tool(
        name=f"confirm_{tool.name}_facts",
        description=(
            f"Use after the resident provides a profile and asks to run {tool.name}. "
            f"{tool.name} is enabled but hidden until this review is approved. "
            "Once its required fields are supported by the conversation, use this "
            "confirmation immediately. Do not delay for optional fields; omit unknown "
            "optional values. Include only exact resident-provided or confirmed facts. "
            "This opens the exact structured facts for resident approval and runs the "
            "requested read-only check after approval."
        ),
        parameters=tool.parameters,
        handler=confirm,
        requires_approval=True,
        title=f"Confirm resident facts for {tool.title or tool.name}",
        module=tool.module,
    )


def build_module_capabilities(
    registry: Registry,
    tools: dict[str, Tool],
) -> tuple[list[PydanticTool], list[Capability[ToolContext]]]:
    """Derive deferred runtime capabilities from authoritative module manifests."""
    modules = {module.name: module for module in registry.modules}

    def root_name(module_name: str) -> str:
        seen: set[str] = set()
        module = modules[module_name]
        while module.parent and module.parent in modules and module.name not in seen:
            seen.add(module.name)
            module = modules[module.parent]
        return module.name

    module_tools: dict[str, list[PydanticTool]] = {}
    shared_tools: list[PydanticTool] = []
    for tool in tools.values():
        adapted = adapt_tool(tool)
        if tool.module in modules:
            module_tools.setdefault(root_name(tool.module), []).append(adapted)
        else:
            shared_tools.append(adapted)

    descendants: dict[str, list] = {}
    for module in registry.modules:
        descendants.setdefault(root_name(module.name), []).append(module)

    capabilities: list[Capability[ToolContext]] = []
    for module in registry.modules:
        if module.parent:
            continue
        instructions = "\n\n".join(
            member.prompt
            for member in descendants[module.name]
            if member.prompt.strip()
        )
        available_tools = module_tools.get(module.name, ())
        availability = (
            "Enabled module action tools: "
            + ", ".join(f"`{tool.name}`" for tool in available_tools)
            + ". These are the only module actions currently available. Do not collect "
            "inputs for or claim to perform any other module action. An action absent "
            "from this enabled list is disabled even if earlier instructions describe "
            "it conditionally."
            if available_tools
            else "This capability has no module-specific action tools enabled. Do not "
            "collect inputs for or claim to perform a module action. An action absent "
            "from this enabled list is disabled even if earlier instructions describe "
            "it conditionally."
        )
        instructions = "\n\n".join(part for part in (instructions, availability) if part)
        capabilities.append(
            Capability(
                id=module.name,
                description=module.description or f"NYC {module.category} help",
                instructions=instructions,
                tools=available_tools,
                defer_loading=True,
            )
        )
    return shared_tools, capabilities


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
                and part.tool_name != _GROUNDED_OUTPUT_TOOL
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
                ) and part.tool_name != _GROUNDED_OUTPUT_TOOL:
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
    native = _native_cost(messages)
    if native is not None:
        return native, "pydantic-native"
    fallback = priced_cost_usd(
        model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
    )
    return fallback, "litellm-fallback" if fallback is not None else "unpriced"


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
        "Use only citation IDs returned": "unknown_citation",
        "When a grounded block includes legacy citation markers": "citation_mismatch",
        "Do not write citation markers": "citation_marker",
        "Return a complete replacement answer to the resident's full request": "grounding",
        "Return a complete replacement answer. Keep every supported outcome": "semantic_grounding",
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


class _BoundedMemoryCapability(AbstractCapability[ToolContext]):
    """Adapt HeyNYC memory at PydanticAI's complete-request seam."""

    def __init__(self, conversation: "_PydanticConversation") -> None:
        self.conversation = conversation
        self.visible_history: list[dict] | None = None
        self.compacted = False

    async def before_model_request(
        self,
        ctx: RunContext[ToolContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self.conversation._prepare_model_request(
            request_context,
            self,
        )


class _ModelTimingCapability(AbstractCapability[ToolContext]):
    """Measure only native provider requests, excluding tools and orchestration."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self.request_ms: list[float] = []

    async def wrap_model_request(
        self,
        ctx: RunContext[ToolContext],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        started = time.perf_counter()
        try:
            return await handler(request_context)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.elapsed_ms += elapsed_ms
            self.request_ms.append(round(elapsed_ms, 3))


class PydanticRuntimeAdapter:
    """Run existing HeyNYC tools through PydanticAI without changing production runtime code."""

    def __init__(
        self,
        model: Any,
        *,
        registry: Registry,
        tools: dict[str, Tool],
        system_prompt: str = "",
        prompt_builder: Callable[[str], str] | None = None,
        guard_grounding: bool = True,
        use_module_capabilities: bool = False,
        current_awareness: Callable[[], Awaitable[str]] | None = None,
        extra_capabilities: Sequence[Any] = (),
        usage_limits: UsageLimits | None = None,
        instrument: InstrumentationSettings | None = None,
        context_budget: int | None = None,
        measure_context: MeasureFn | None = None,
        compact_context: CompactFn | None = None,
        answer_model_route: str | None = None,
        structured_grounding: bool = False,
        semantic_verifier: Any = None,
    ) -> None:
        self.registry = registry
        self.tools = dict(tools)
        for tool in tools.values():
            if tool.resident_fact_scope:
                confirmation = resident_fact_confirmation_tool(tool)
                self.tools[confirmation.name] = confirmation
        self.model = getattr(model, "model_name", type(model).__name__)
        self._current_awareness = current_awareness
        self._usage_limits = usage_limits or UsageLimits(request_limit=8)
        self._answer_model_route = answer_model_route
        self._semantic_verifier = semantic_verifier
        self._context_budget = (
            context_capacity(answer_model_route, None, True)
            if context_budget is None and answer_model_route is not None
            else context_budget
        )
        self._measure_context = measure_context
        self._compact_context = compact_context
        adapted_tools, capabilities = (
            build_module_capabilities(registry, self.tools)
            if use_module_capabilities
            else ([adapt_tool(tool) for tool in self.tools.values()], [])
        )
        agent_model = InstrumentedModel(model, instrument) if instrument else model
        if structured_grounding:
            system_prompt = "\n\n".join(filter(None, (
                system_prompt,
                _STRUCTURED_GROUNDING_SYSTEM_PROMPT,
            )))
        self._agent = PydanticAgent(
            agent_model,
            deps_type=ToolContext,
            tools=adapted_tools,
            capabilities=[
                ReinjectSystemPrompt(),
                *capabilities,
                *extra_capabilities,
            ],
            system_prompt=system_prompt,
            model_settings=_native_cache_settings(model),
            tool_timeout=30,
            output_type=[
                (
                    ToolOutput(
                        GroundedAnswer,
                        name=_GROUNDED_OUTPUT_TOOL,
                        description=(
                            "Answer the resident's actual question first, in the first "
                            "grounded block, not in the acknowledgment. If the resident's "
                            "individual outcome cannot be determined from retrieved "
                            "evidence, state that limitation in the acknowledgment, then "
                            "give the supported general guidance and next step. Return each "
                            "resident-facing factual or procedural claim as an atomic "
                            "grounded block with retrieved citation IDs. If the evidence "
                            "gives only general rules, plainly distinguish them from the "
                            "resident's individual outcome. When the resident asks what "
                            "will happen or how to protect or access a service, include a "
                            "concrete official next step supported by a retrieved source."
                        ),
                    )
                    if structured_grounding
                    else str
                ),
                DeferredToolRequests,
            ],
            retries={"tools": 1, "output": 2},
        )
        if prompt_builder is not None:
            self._agent.instructions(lambda ctx: prompt_builder(ctx.deps.query))
        if guard_grounding:
            self._agent.output_validator(self._validate_grounding)

    async def _validate_grounding(
        self,
        ctx: RunContext[ToolContext],
        output: str | GroundedAnswer | DeferredToolRequests,
    ) -> str | GroundedAnswer | DeferredToolRequests:
        if isinstance(output, DeferredToolRequests):
            return output
        if isinstance(output, GroundedAnswer):
            if not output.grounded_blocks:
                raise ModelRetry(
                    "Answer with at least one grounded block supported by a retrieved source."
                )
            mapping = ctx.deps.citations.mapping()
            unknown = sorted({
                citation_id
                for block in output.grounded_blocks
                for citation_id in block.citation_ids
                if citation_id not in mapping
            })
            if unknown:
                raise ModelRetry(
                    "Use only citation IDs returned by tools in this run."
                )
            for block in output.grounded_blocks:
                embedded = set(used_citations(block.text, mapping))
                declared = set(block.citation_ids)
                if embedded and embedded != declared:
                    raise ModelRetry(
                        "When a grounded block includes legacy citation markers, those "
                        "markers must exactly match citation_ids."
                    )
            authored = [
                output.acknowledgment,
                output.follow_up_question,
                *(_grounded_block_text(block) for block in output.grounded_blocks),
            ]
            if any("{cite:" in text for text in authored):
                raise ModelRetry(
                    "Do not write citation markers. Put source IDs in citation_ids; "
                    "the runtime renders markers."
                )
            rendered = _render_grounded_answer(output)
        else:
            rendered = output
        verdict = check_grounding(
            rendered,
            ctx.deps.citations.mapping(),
            ctx.deps.query,
        )
        if verdict is not None and verdict.blocking:
            raise ModelRetry(
                "Return a complete replacement answer to the resident's full request, "
                "not a correction or addendum. Preserve all still-supported requested "
                "outcomes from prior tool results, omit unsupported details, and cite every "
                "factual claim. A deterministic grounding check rejected at least one claim."
            )
        if isinstance(output, GroundedAnswer) and self._semantic_verifier is not None:
            mapping = ctx.deps.citations.mapping()
            inputs = []
            if output.acknowledgment.strip():
                inputs.append(NLIInput(
                    id="acknowledgment",
                    claim=output.acknowledgment,
                    source="",
                    kind="framing",
                ))
            inputs.extend(
                NLIInput(
                    id=f"block-{index}",
                    claim=_grounded_block_text(block),
                    source="\n\n".join(
                        f"[{citation_id}] "
                        f"{_semantic_citation_evidence(mapping[citation_id])}"
                        for citation_id in block.citation_ids
                    )[:_SEMANTIC_EVIDENCE_CHARS],
                )
                for index, block in enumerate(output.grounded_blocks)
            )
            if output.follow_up_question.strip():
                inputs.append(NLIInput(
                    id="follow-up-question",
                    claim=output.follow_up_question,
                    source="",
                    kind="question",
                ))
            semantic = await self._semantic_verifier.arun_many(inputs)
            ctx.deps.semantic_verifier_runs.append({
                "input_tokens": semantic.input_tokens,
                "output_tokens": semantic.output_tokens,
                "cached_input_tokens": semantic.cached_input_tokens,
                "cost_usd": semantic.cost_usd,
                "latency_ms": semantic.latency_ms,
                "error": semantic.error,
                "labels": [verdict.label for verdict in semantic.verdicts],
                "items": [
                    {
                        "position": position,
                        "kind": item.kind,
                        "label": verdict.label,
                    }
                    for position, (item, verdict) in enumerate(
                        zip(inputs, semantic.verdicts, strict=True)
                    )
                ],
            })
            if semantic.error is not None:
                return GroundedAnswer(
                    acknowledgment=(
                        "I'm sorry, I couldn't verify the sources needed to answer safely "
                        "right now. Please try again."
                    )
                )
            if any(not verdict.supported for verdict in semantic.verdicts):
                raise ModelRetry(
                    "Return a complete replacement answer. Keep every supported outcome, "
                    "but remove or narrow claims that the cited evidence does not support. "
                    "Each grounded block must be one claim wholly supported by its cited "
                    "evidence. Remove unsupported conditions and conclusions, keep neutral "
                    "clarification in follow_up_question, and do not add uncited procedural "
                    "advice."
                )
        if feedback := _reply_script_feedback(ctx.deps.query, rendered):
            raise ModelRetry(feedback)
        return output

    @staticmethod
    def _merge_semantic_usage(result: AgentResult, runs: list[dict[str, Any]]) -> None:
        if not runs:
            return
        input_tokens = sum(int(run["input_tokens"]) for run in runs)
        output_tokens = sum(int(run["output_tokens"]) for run in runs)
        cached_tokens = sum(int(run["cached_input_tokens"]) for run in runs)
        costs = [run.get("cost_usd") for run in runs]
        semantic_cost = (
            sum(float(cost) for cost in costs)
            if all(isinstance(cost, (int, float)) for cost in costs)
            else None
        )
        result.usage.update({
            "semantic_verifier_requests": len(runs),
            "semantic_verifier_input_tokens": input_tokens,
            "semantic_verifier_output_tokens": output_tokens,
            "semantic_verifier_cached_input_tokens": cached_tokens,
            "semantic_verifier_cost_usd": semantic_cost,
            "semantic_verifier_time_ms": sum(float(run["latency_ms"]) for run in runs),
        })
        if errors := [run["error"] for run in runs if run.get("error")]:
            result.usage["semantic_verifier_error"] = errors[-1]
        labels: dict[str, int] = {}
        for run in runs:
            for label in run["labels"]:
                labels[label] = labels.get(label, 0) + 1
        result.usage["semantic_verifier_labels"] = labels
        result.usage["input_tokens"] += input_tokens
        result.usage["output_tokens"] += output_tokens
        result.usage["cached_input_tokens"] += cached_tokens
        result.usage["requests"] += len(runs)
        result.usage["n_model_calls"] += len(runs)
        answer_cost = result.usage.get("cost_usd")
        result.usage["cost_usd"] = (
            float(answer_cost) + semantic_cost
            if isinstance(answer_cost, (int, float)) and semantic_cost is not None
            else None
        )
        result.usage["cost_status"] = (
            "priced" if result.usage["cost_usd"] is not None else "unpriced"
        )

    def conversation(self) -> "_PydanticConversation":
        return _PydanticConversation(self)

    def _failed_result(
        self,
        messages: Sequence[ModelMessage],
        *,
        citations: CitationRegistry,
        started: float,
        timing_capability: _ModelTimingCapability,
        semantic_verifier_runs: list[dict[str, Any]],
        status: str,
    ) -> AgentResult:
        result = self._project_result(
            messages,
            _captured_usage(messages),
            TEMPORARY_FAILURE_FALLBACK,
            citations,
            started,
            model_time_ms=timing_capability.elapsed_ms,
            status=status,
        )
        result.usage["model_request_ms"] = timing_capability.request_ms
        result.usage["retry_kinds"] = _retry_kinds(messages)
        self._merge_semantic_usage(result, semantic_verifier_runs)
        return result

    def conversation_from_state(self, state: bytes) -> "_PydanticConversation":
        return _PydanticConversation.from_state(self, state)

    def conversation_from_transcript(
        self,
        transcript: Sequence[dict],
    ) -> "_PydanticConversation":
        conversation = self.conversation()
        conversation._history = _native_history(transcript)
        conversation._user_turns = tuple(
            str(turn.get("content") or "")
            for turn in transcript
            if turn.get("role") == "user"
        )
        return conversation

    async def run(
        self,
        user_message: str,
        *,
        reminders: list[str] | None = None,
        output_dir: Path | None = None,
        drafts: Any = None,
        resident_facts: dict[str, ResidentFact] | None = None,
        event_sink: Callable[[events.Event], None] | None = None,
        **_: Any,
    ) -> AgentResult:
        return await self.conversation().send(
            user_message,
            reminders=reminders,
            output_dir=output_dir,
            drafts=drafts,
            resident_facts=resident_facts,
            event_sink=event_sink,
        )

    async def _run(
        self,
        user_message: str,
        *,
        message_history: Sequence[ModelMessage],
        prior_user_turns: tuple[str, ...],
        reminders: list[str] | None,
        output_dir: Path | None,
        drafts: Any,
        resident_facts: dict[str, ResidentFact] | None,
        timing_capability: _ModelTimingCapability,
        event_sink: Callable[[events.Event], None] | None = None,
        citations: CitationRegistry | None = None,
        memory_capability: _BoundedMemoryCapability | None = None,
    ) -> tuple[
        AgentResult,
        list[ModelMessage],
        DeferredToolRequests | None,
    ]:
        citations = citations if citations is not None else CitationRegistry()
        user_turns = (*prior_user_turns, user_message)
        started = time.perf_counter()
        message_id = f"pydantic-{time.monotonic_ns()}"
        _emit(
            event_sink,
            events.SessionInit(session_id=message_id, model=self.model),
        )
        _emit(event_sink, events.MessageStart(message_id=message_id))
        backstop = (
            _emergency_backstop(user_message)
            or _sensitive_identifier_backstop(user_message)
            or _internal_config_backstop(user_message)
        )
        if backstop is not None:
            new_messages: list[ModelMessage] = [
                ModelRequest(parts=[UserPromptPart(user_message)]),
                ModelResponse(parts=[TextPart(backstop)]),
            ]
            result = AgentResult(
                    text=backstop,
                    citations=citations.mapping(),
                    tool_calls_made=[],
                    iterations=0,
                    status="success",
                    messages=_openai_messages(new_messages),
                    usage={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "answer_input_tokens": 0,
                        "answer_output_tokens": 0,
                        "answer_cached_input_tokens": 0,
                        "requests": 0,
                        "tool_calls": 0,
                        "n_model_calls": 0,
                        "n_answer_model_calls": 0,
                        "n_tool_calls": 0,
                        "iterations": 0,
                        "capabilities_used": [],
                        "cost_usd": 0.0,
                        "cost_status": "priced",
                        "cost_source": "deterministic",
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    },
            )
            _emit(
                event_sink,
                events.TextDelta(message_id=message_id, text=backstop),
            )
            _finish_events(event_sink, message_id, result)
            return result, new_messages, None
        deps = ToolContext(
            citations=citations,
            registry=self.registry,
            query=user_message,
            user_history="\n".join(user_turns),
            user_turns=user_turns,
            toolbox=self.tools,
            output_dir=output_dir,
            drafts=drafts,
            resident_facts=resident_facts if resident_facts is not None else {},
        )
        instructions = list(reminders or ())
        if self._current_awareness is not None:
            awareness = await self._current_awareness()
            if awareness:
                instructions.append(awareness)
        try:
            with capture_run_messages() as captured:
                native = await self._agent.run(
                    user_message,
                    message_history=message_history or None,
                    instructions=_dynamic_instructions(instructions),
                    deps=deps,
                    usage_limits=self._usage_limits,
                    event_stream_handler=(
                        (
                            lambda ctx, stream: _forward_events(
                                event_sink,
                                message_id,
                                stream,
                            )
                        )
                        if event_sink is not None
                        else None
                    ),
                    capabilities=(
                        [
                            timing_capability,
                            *(
                                [memory_capability]
                                if memory_capability is not None
                                else []
                            ),
                        ]
                    ),
                )
        except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
            current_index = max(
                (
                    index
                    for index, message in enumerate(captured)
                    if isinstance(message, ModelRequest)
                    and any(
                        isinstance(part, UserPromptPart)
                        and part.content == user_message
                        for part in message.parts
                    )
                ),
                default=len(message_history),
            )
            new_messages = captured[current_index:]
            result = self._failed_result(
                new_messages,
                citations=citations,
                started=started,
                timing_capability=timing_capability,
                semantic_verifier_runs=deps.semantic_verifier_runs,
                status=(
                    "max_turns"
                    if isinstance(exc, UsageLimitExceeded)
                    else "error"
                ),
            )
            if isinstance(exc, UnexpectedModelBehavior):
                _finish_events(event_sink, message_id, result)
                raise PydanticRunFailure(
                    exc.message,
                    result,
                    {"semantic_verifier_runs": deps.semantic_verifier_runs},
                ) from exc
            result.hit_max_iters = True
            _finish_events(event_sink, message_id, result)
            return result, new_messages, None
        result = self._result(
            native,
            citations,
            started,
            model_time_ms=timing_capability.elapsed_ms,
        )
        result.usage["model_request_ms"] = timing_capability.request_ms
        self._merge_semantic_usage(result, deps.semantic_verifier_runs)
        _finish_events(event_sink, message_id, result)
        pending = (
            native.output if isinstance(native.output, DeferredToolRequests) else None
        )
        return result, native.new_messages(), pending

    def _result(
        self,
        native: Any,
        citations: CitationRegistry,
        started: float,
        *,
        model_time_ms: float,
    ) -> AgentResult:
        return self._project_result(
            native.new_messages(),
            native.usage,
            native.output,
            citations,
            started,
            model_time_ms=model_time_ms,
        )

    def _project_result(
        self,
        new_messages: Sequence[ModelMessage],
        usage: RunUsage,
        output: str | GroundedAnswer | DeferredToolRequests,
        citations: CitationRegistry,
        started: float,
        *,
        model_time_ms: float,
        status: str | None = None,
    ) -> AgentResult:
        tool_calls = [
            part.tool_name
            for message in new_messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
            and part.tool_name != _GROUNDED_OUTPUT_TOOL
        ]
        capabilities_used = list(
            dict.fromkeys(
                str(part.args_as_dict()["id"])
                for message in new_messages
                if isinstance(message, ModelResponse)
                for part in message.parts
                if isinstance(part, ToolCallPart)
                and part.tool_name == "load_capability"
                and "id" in part.args_as_dict()
            )
        )
        executed_tool_calls = [
            part.tool_name
            for message in new_messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
            and part.tool_name != _GROUNDED_OUTPUT_TOOL
        ]
        pending = isinstance(output, DeferredToolRequests)
        iterations = sum(isinstance(message, ModelResponse) for message in new_messages)
        cost, cost_source = _complete_cost(self.model, new_messages, usage)
        text = ""
        if not pending:
            text = attach_temporal_provenance(
                (
                    _render_grounded_answer(output)
                    if isinstance(output, GroundedAnswer)
                    else str(output)
                ),
                citations.mapping(),
            )
        return AgentResult(
            text=text,
            citations=citations.mapping(),
            tool_calls_made=tool_calls,
            iterations=iterations,
            status=status or ("approval_required" if pending else "success"),
            messages=_openai_messages(new_messages),
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cache_read_tokens,
                "answer_input_tokens": usage.input_tokens,
                "answer_output_tokens": usage.output_tokens,
                "answer_cached_input_tokens": usage.cache_read_tokens,
                "requests": usage.requests,
                "tool_calls": usage.tool_calls,
                "n_model_calls": usage.requests,
                "n_answer_model_calls": usage.requests,
                "n_tool_calls": usage.tool_calls,
                "executed_tool_calls": executed_tool_calls,
                "iterations": iterations,
                "capabilities_used": capabilities_used,
                "cost_usd": cost,
                "cost_status": "priced" if cost is not None else "unpriced",
                "cost_source": cost_source,
                "model_time_ms": model_time_ms,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )


class _PydanticConversation:
    def __init__(self, runtime: PydanticRuntimeAdapter) -> None:
        self.runtime = runtime
        self._history: list[ModelMessage] = []
        self._user_turns: tuple[str, ...] = ()
        self._pending: DeferredToolRequests | None = None
        self._resident_facts: dict[str, ResidentFact] = {}
        self._citations = CitationRegistry()
        self.continuity: ContinuityRecord | None = None
        self._memory_usage: dict = {}
        self._memory_spend = SpendGuard(config.HEYNYC_SPEND_CAP)

    @classmethod
    def from_state(
        cls,
        runtime: PydanticRuntimeAdapter,
        state: bytes,
    ) -> "_PydanticConversation":
        payload = json.loads(state)
        conversation = cls(runtime)
        conversation._history = ModelMessagesTypeAdapter.validate_python(
            payload["messages"]
        )
        conversation._user_turns = tuple(payload["user_turns"])
        conversation._resident_facts = _RESIDENT_FACTS.validate_python(
            payload.get("resident_facts", {})
        )
        if continuity := payload.get("continuity"):
            conversation.continuity = ContinuityRecord.model_validate(continuity)
        if citations := payload.get("citations") or payload.get("pending_citations"):
            conversation._citations = CitationRegistry.from_state(citations)
        if payload["pending"] is not None:
            conversation._pending = _DEFERRED_REQUESTS.validate_python(
                payload["pending"]
            )
            conversation._validate_pending_history()
        return conversation

    def _validate_pending_history(self) -> None:
        if self._pending is None:
            return
        history_calls = {
            part.tool_call_id: (part.tool_name, part.args_as_dict())
            for message in self._history
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        for call in (*self._pending.approvals, *self._pending.calls):
            expected = (call.tool_name, call.args_as_dict())
            if history_calls.get(call.tool_call_id) != expected:
                raise ValueError(
                    f"Deferred call {call.tool_call_id!r} does not match message history"
                )

    @property
    def pending_approvals(self) -> dict[str, dict]:
        if self._pending is None:
            return {}
        return {
            call.tool_call_id: {
                "tool_name": call.tool_name,
                "args": call.args_as_dict(),
            }
            for call in self._pending.approvals
        }

    @property
    def pending_calls(self) -> dict[str, dict]:
        if self._pending is None:
            return {}
        return {
            call.tool_call_id: {
                "tool_name": call.tool_name,
                "args": call.args_as_dict(),
            }
            for call in self._pending.calls
        }

    def dump_state(self) -> bytes:
        """Serialize native state; the caller must use authenticated encrypted storage."""
        return json.dumps(
            {
                "messages": ModelMessagesTypeAdapter.dump_python(
                    self._history,
                    mode="json",
                ),
                "user_turns": self._user_turns,
                "resident_facts": _RESIDENT_FACTS.dump_python(
                    self._resident_facts,
                    mode="json",
                ),
                "continuity": (
                    self.continuity.model_dump(mode="json")
                    if self.continuity is not None
                    else None
                ),
                "pending": (
                    _DEFERRED_REQUESTS.dump_python(self._pending, mode="json")
                    if self._pending is not None
                    else None
                ),
                "citations": self._citations.dump_state(),
            },
            separators=(",", ":"),
        ).encode()

    async def _prepare_model_request(
        self,
        request_context: ModelRequestContext,
        capability: _BoundedMemoryCapability,
    ) -> ModelRequestContext:
        messages = request_context.messages
        current_index = max(
            (
                index
                for index, message in enumerate(messages)
                if any(isinstance(part, UserPromptPart) for part in message.parts)
            ),
            default=len(messages),
        )
        current = messages[current_index:]
        system_parts = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, SystemPromptPart)
            and part.content
        ]
        if (
            self.runtime._answer_model_route is None
            and (
                self.runtime._measure_context is None
                or self.runtime._compact_context is None
            )
        ):
            return request_context
        schemas = _function_tool_schemas(request_context)

        def measure_complete(
            history: list[dict],
            continuity: ContinuityRecord | None,
        ) -> int:
            projected = [
                *_native_history(history),
                *_native_orchestration_history(messages[:current_index]),
                *current,
            ]
            measured = _measurement_messages(
                projected,
                omit_instruction=(
                    continuity_reminder(continuity) if continuity is not None else ""
                ),
            )
            if system_parts and not any(
                isinstance(part, SystemPromptPart)
                for message in projected
                for part in message.parts
            ):
                measured[:0] = [
                    {"role": "system", "content": part.content}
                    for part in system_parts
                ]
            if self.runtime._measure_context is not None:
                return self.runtime._measure_context(measured, continuity)
            assert self.runtime._answer_model_route is not None
            reminder = (
                continuity_reminder(continuity)
                if continuity is not None
                else ""
            )
            if reminder and not any(
                reminder in str(message.get("content") or "")
                for message in measured
            ):
                measured.append({"role": "system", "content": reminder})
            return request_tokens(
                self.runtime._answer_model_route,
                measured,
                schemas,
            )

        async def compact(
            older: list[dict],
            current_continuity: ContinuityRecord | None,
        ) -> ContinuityRecord | dict:
            try:
                if capability.compacted:
                    return current_continuity or ContinuityRecord()
                if self.runtime._compact_context is not None:
                    record = await self.runtime._compact_context(
                        older,
                        current_continuity,
                    )
                else:
                    record, usage = await compact_memory(
                        older,
                        current_continuity,
                        self._memory_spend,
                    )
                    self._memory_usage.update(usage)
                capability.compacted = True
                return record
            except ContextCapacityError:
                raise
            except Exception as exc:
                raise ContextCapacityError(
                    "continuity compaction is unavailable"
                ) from exc

        plan = await prepare_context(
            (
                capability.visible_history
                if capability.visible_history is not None
                else _resident_history(messages[:current_index])
            ),
            self.continuity,
            budget=self.runtime._context_budget,
            measure=measure_complete,
            compact=compact,
        )
        capability.visible_history = plan.history
        self.continuity = plan.continuity
        if plan.compacted or not self._memory_usage:
            self._memory_usage.update({
                "memory_compactions": int(plan.compacted),
                "memory_pre_tokens": plan.pre_compaction_tokens,
                "memory_post_tokens": plan.post_compaction_tokens,
            })
        if self.continuity is not None:
            request = next(
                (
                    message
                    for message in current
                    if isinstance(message, ModelRequest)
                    and any(
                        isinstance(part, UserPromptPart) for part in message.parts
                    )
                ),
                None,
            )
            if request is not None:
                reminder = continuity_reminder(self.continuity)
                if reminder not in (request.instructions or ""):
                    request.instructions = "\n\n".join(
                        part for part in (request.instructions, reminder) if part
                    )
        processed = [
            *_native_history(plan.history),
            *_native_orchestration_history(messages[:current_index]),
            *current,
        ]
        if system_parts and not any(
            isinstance(part, SystemPromptPart)
            for message in processed
            for part in message.parts
        ):
            request = next(
                message for message in processed if isinstance(message, ModelRequest)
            )
            request.parts[:0] = system_parts
        request_context.messages = processed
        return request_context

    async def send(
        self,
        user_message: str,
        *,
        reminders: list[str] | None = None,
        output_dir: Path | None = None,
        drafts: Any = None,
        resident_facts: dict[str, ResidentFact] | None = None,
        event_sink: Callable[[events.Event], None] | None = None,
        **_: Any,
    ) -> AgentResult:
        if self._pending is not None:
            raise ValueError("Cannot start a new turn while approval is pending")
        if resident_facts:
            self._resident_facts.update(resident_facts)
        self._memory_usage.clear()
        memory_capability = (
            _BoundedMemoryCapability(self)
            if (
                self.runtime._answer_model_route is not None
                or (
                    self.runtime._measure_context is not None
                    and self.runtime._compact_context is not None
                )
            )
            else None
        )
        timing_capability = _ModelTimingCapability()
        try:
            result, new_messages, self._pending = await self.runtime._run(
                user_message,
                message_history=self._history,
                prior_user_turns=self._user_turns,
                reminders=reminders,
                output_dir=output_dir,
                drafts=drafts,
                resident_facts=self._resident_facts,
                citations=self._citations,
                memory_capability=memory_capability,
                timing_capability=timing_capability,
                event_sink=event_sink,
            )
            merge_memory_usage(
                result.usage,
                self._memory_usage,
                latency_already_included=True,
            )
        finally:
            self._memory_usage.clear()
        self._history.extend(new_messages)
        self._user_turns = (*self._user_turns, user_message)
        return result

    async def resume_approvals(
        self,
        approvals: dict[str, bool],
        *,
        output_dir: Path | None = None,
        drafts: Any = None,
        event_sink: Callable[[events.Event], None] | None = None,
    ) -> AgentResult:
        if self._pending is None:
            raise ValueError("No deferred approval is pending")
        self._validate_pending_history()
        expected = set(self.pending_approvals)
        if set(approvals) != expected:
            raise ValueError(
                f"Approval IDs must match pending calls: {sorted(expected)}"
            )
        query = self._user_turns[-1] if self._user_turns else ""
        citations = self._citations
        deps = ToolContext(
            citations=citations,
            registry=self.runtime.registry,
            query=query,
            user_history="\n".join(self._user_turns),
            user_turns=self._user_turns,
            toolbox=self.runtime.tools,
            output_dir=output_dir,
            drafts=drafts,
            resident_facts=self._resident_facts,
        )
        started = time.perf_counter()
        message_id = f"pydantic-{time.monotonic_ns()}"
        _emit(
            event_sink,
            events.SessionInit(session_id=message_id, model=self.runtime.model),
        )
        _emit(event_sink, events.MessageStart(message_id=message_id))
        timing_capability = _ModelTimingCapability()
        try:
            with capture_run_messages() as captured:
                native = await self.runtime._agent.run(
                    message_history=self._history,
                    deferred_tool_results=DeferredToolResults(approvals=approvals),
                    deps=deps,
                    capabilities=[timing_capability],
                    usage_limits=self.runtime._usage_limits,
                    event_stream_handler=(
                        (
                            lambda ctx, stream: _forward_events(
                                event_sink,
                                message_id,
                                stream,
                            )
                        )
                        if event_sink is not None
                        else None
                    ),
                )
        except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
            new_messages = captured[len(self._history):]
            self._history.extend(new_messages)
            self._pending = None
            result = self.runtime._failed_result(
                new_messages,
                citations=citations,
                started=started,
                timing_capability=timing_capability,
                semantic_verifier_runs=deps.semantic_verifier_runs,
                status=(
                    "max_turns"
                    if isinstance(exc, UsageLimitExceeded)
                    else "error"
                ),
            )
            if isinstance(exc, UnexpectedModelBehavior):
                _finish_events(event_sink, message_id, result)
                raise PydanticRunFailure(
                    exc.message,
                    result,
                    {"semantic_verifier_runs": deps.semantic_verifier_runs},
                ) from exc
            result.hit_max_iters = True
            _finish_events(event_sink, message_id, result)
            return result
        result = self.runtime._result(
            native,
            citations,
            started,
            model_time_ms=timing_capability.elapsed_ms,
        )
        result.usage["model_request_ms"] = timing_capability.request_ms
        self.runtime._merge_semantic_usage(result, deps.semantic_verifier_runs)
        _finish_events(event_sink, message_id, result)
        self._history.extend(native.new_messages())
        self._pending = (
            native.output if isinstance(native.output, DeferredToolRequests) else None
        )
        return result


class PydanticApprovalFlow:
    """Persist and resume native Pydantic approvals through the shared encrypted store."""

    def __init__(
        self,
        runtime: PydanticRuntimeAdapter,
        store: Any,
        user_key: str,
        *,
        ttl_s: float,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.user_key = user_key
        self.ttl_s = ttl_s
        state = store.get_pending_approval(user_key)
        self.conversation = (
            runtime.conversation_from_state(state)
            if state is not None
            else runtime.conversation()
        )

    async def send(self, user_message: str, **kwargs: Any) -> AgentResult:
        if self.store.has_pending_approval(self.user_key):
            raise ValueError("Cannot start a new turn while approval is pending")
        result = await self.conversation.send(user_message, **kwargs)
        self._persist_if_pending(result)
        return result

    async def resume(
        self,
        decision: bool | dict[str, bool],
        **kwargs: Any,
    ) -> AgentResult:
        expected = set(self.conversation.pending_approvals)
        if not isinstance(decision, bool) and set(decision) != expected:
            raise ValueError(
                f"Approval IDs must match pending calls: {sorted(expected)}"
            )
        if not isinstance(decision, bool) and not all(
            isinstance(value, bool) for value in decision.values()
        ):
            raise ValueError("Approval decisions must be booleans")
        decisions = (
            {
                call_id: decision
                for call_id in self.conversation.pending_approvals
            }
            if isinstance(decision, bool)
            else decision
        )
        retry_safe = all(
            not approved
            or request["tool_name"].startswith("confirm_")
            and request["tool_name"].endswith("_facts")
            or (
                (tool := self.runtime.tools.get(request["tool_name"])) is not None
                and tool.idempotent
            )
            for call_id, request in self.conversation.pending_approvals.items()
            for approved in (decisions[call_id],)
        )
        state = (
            self.store.get_pending_approval(self.user_key)
            if retry_safe
            else self.store.pop_pending_approval(self.user_key)
        )
        if state is None:
            raise ValueError("Pending approval expired or already consumed")
        self.conversation = self.runtime.conversation_from_state(state)
        incomplete = False
        try:
            result = await self.conversation.resume_approvals(decisions, **kwargs)
        except PydanticRunFailure as exc:
            result = exc.partial_result
            incomplete = True
        incomplete = incomplete or result.status in {"error", "max_turns"}
        if retry_safe:
            if incomplete:
                self.conversation = self.runtime.conversation_from_state(state)
            else:
                self.store.pop_pending_approval(self.user_key)
        self._persist_if_pending(result)
        return result

    def _persist_if_pending(self, result: AgentResult) -> None:
        if result.status == "approval_required":
            if self.conversation.pending_calls:
                raise ValueError(
                    "External deferred calls are not supported by this approval flow"
                )
            self.store.set_pending_approval(
                self.user_key,
                self.conversation.dump_state(),
                ttl_s=self.ttl_s,
            )
            result.text = self.review_text()

    def review_text(self) -> str:
        return approval_review_text(self.conversation.pending_approvals)


def approval_review_text(pending_approvals: dict[str, dict]) -> str:
    requests = tuple(pending_approvals.values())
    copies = tuple(_approval_copy(request["tool_name"]) for request in requests)
    mixed = len(set(copies)) > 1
    heading, question = (
        (
            "Review each item below:",
            "Reply YES to confirm all facts and approve all actions, "
            "or NO to correct or deny them.",
        )
        if mixed
        else copies[0]
    )
    lines = [heading]
    for request, (item_heading, _) in zip(requests, copies, strict=True):
        arguments = json.dumps(
            request["args"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if mixed:
            lines.extend(("", item_heading))
        lines.extend(("", request["tool_name"], arguments))
    lines.extend(("", question))
    return "\n".join(lines)


def _approval_copy(tool_name: str) -> tuple[str, str]:
    if tool_name.startswith("confirm_") and tool_name.endswith("_facts"):
        return (
            "Review the structured facts I understood:",
            "Reply YES if these facts are accurate and run the requested read-only "
            "check, or NO to correct them.",
        )
    return (
        "Review the proposed action and exact values:",
        "Reply YES to approve, or NO to deny.",
    )


def build_runtime(
    registry: Registry,
    *,
    model: Any,
    tools: dict[str, Tool] | None = None,
    index: Any = None,
    use_module_capabilities: bool = False,
    current_awareness: Callable[[], Awaitable[str]] | None = None,
    extra_capabilities: Sequence[Any] = (),
    answer_model_route: str | None = None,
    structured_grounding: bool = False,
    semantic_verifier: Any = None,
) -> PydanticRuntimeAdapter:
    """Build the isolated parity runtime around a caller-selected Pydantic model."""
    stable_prompt, _ = build_system_prompt_tiers(
        registry,
        query="",
        include_module_guidance=not use_module_capabilities,
    )
    return PydanticRuntimeAdapter(
        model,
        registry=registry,
        tools=tools if tools is not None else build_toolbox(registry, index=index),
        system_prompt=stable_prompt,
        prompt_builder=lambda query: build_system_prompt_tiers(
            registry,
            query=query,
            include_module_guidance=not use_module_capabilities,
        )[1],
        use_module_capabilities=use_module_capabilities,
        current_awareness=current_awareness,
        extra_capabilities=extra_capabilities,
        answer_model_route=answer_model_route,
        structured_grounding=structured_grounding,
        semantic_verifier=semantic_verifier,
    )


def _uses_openai_responses(model: str, *, has_tools: bool = True) -> bool:
    if not model.startswith("openai/"):
        return False
    model_info, _ = responses_api_bridge_check(
        model.removeprefix("openai/"),
        "openai",
        tools=[{}] if has_tools else [],
        reasoning_effort=config.HEYNYC_REASONING_EFFORT,
    )
    return model_info.get("mode") == "responses"


def configured_model(model: str) -> Any:
    if model.startswith("openai/"):
        settings = {
            key: value
            for key, value in {
                "openai_reasoning_effort": config.HEYNYC_REASONING_EFFORT,
                "openai_service_tier": config.HEYNYC_SERVICE_TIER,
            }.items()
            if value is not None
        }
        model_type = OpenAIResponsesModel if _uses_openai_responses(model) else OpenAIChatModel
        return model_type(model.removeprefix("openai/"), settings=settings)
    return infer_model(model.replace("/", ":", 1))


def build_configured_runtime(
    registry: Registry,
    *,
    model: str,
    index: Any = None,
    current_awareness: Callable[[], Awaitable[str]] | None = None,
) -> PydanticRuntimeAdapter:
    return build_runtime(
        registry,
        model=configured_model(model),
        tools=build_toolbox(registry, index=index),
        use_module_capabilities=True,
        current_awareness=current_awareness,
        answer_model_route=model,
    )
