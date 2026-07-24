from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator
from pydantic import TypeAdapter
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
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
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.tools import Tool as PydanticTool
from pydantic_ai.usage import RunUsage

from heynyc.core import config
from heynyc.core.agent import (
    EMPTY_ANSWER_FALLBACK,
    AgentResult,
    _emergency_backstop,
    _internal_config_backstop,
    _reply_script_feedback,
    _sensitive_identifier_backstop,
)
from heynyc.core.citations import CitationRegistry
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
from heynyc.core.prompts import build_system_prompt_tiers
from heynyc.core.registry import Registry
from heynyc.core.spend import SpendGuard
from heynyc.core.telemetry import priced_cost_usd
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.base import ResidentFact, Tool, ToolContext

_DEFERRED_REQUESTS = TypeAdapter(DeferredToolRequests)
_RESIDENT_FACTS = TypeAdapter(dict[str, ResidentFact])


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
        return f"Resident confirmed the structured facts for {tool.name}."

    return Tool(
        name=f"confirm_{tool.name}_facts",
        description=(
            f"Ask the resident to confirm the exact structured profile before {tool.name}. "
            "This records facts only and does not call an external service."
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
            "inputs for or claim to perform any other module action."
            if available_tools
            else "This capability has no module-specific action tools enabled. Do not "
            "collect inputs for or claim to perform a module action."
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


def _openai_messages(messages: Sequence[ModelMessage]) -> list[dict]:
    """Translate native PydanticAI messages into HeyNYC's existing trace contract."""
    translated: list[dict] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = "".join(
                part.content for part in message.parts if isinstance(part, TextPart)
            )
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
                ):
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
        translated.extend(_openai_messages([message]))
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


def _native_cache_settings(model: Any) -> dict[str, Any] | None:
    """Use each provider's PydanticAI-native prompt cache controls."""
    system = getattr(model, "system", "")
    if system == "openai":
        return {"openai_prompt_cache_key": "heynyc-pydantic-v1"}
    if system == "anthropic":
        return {"anthropic_cache_instructions": True}
    return None


def _dynamic_instructions(parts: Sequence[str]) -> Callable[[], str] | None:
    """Keep per-turn context after provider prompt-cache boundaries."""
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return (lambda: text) if text else None


def _resident_history(messages: Sequence[ModelMessage]) -> list[dict]:
    """Collapse native traces to complete resident/final-assistant exchanges."""
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
            text = "".join(
                part.content for part in message.parts if isinstance(part, TextPart)
            )
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
            self.elapsed_ms += (time.perf_counter() - started) * 1000.0


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
            output_type=[str, DeferredToolRequests],
            retries={"tools": 1, "output": 2},
        )
        if prompt_builder is not None:
            self._agent.instructions(lambda ctx: prompt_builder(ctx.deps.query))
        if guard_grounding:
            self._agent.output_validator(self._validate_grounding)

    @staticmethod
    def _validate_grounding(
        ctx: RunContext[ToolContext],
        output: str | DeferredToolRequests,
    ) -> str | DeferredToolRequests:
        if not isinstance(output, str):
            return output
        verdict = check_grounding(output, ctx.deps.citations.mapping(), ctx.deps.query)
        if verdict is not None and verdict.blocking:
            raise ModelRetry(
                "Return a complete replacement answer to the resident's full request, "
                "not a correction or addendum. Preserve all still-supported requested "
                "outcomes from prior tool results, fix the grounding failure below, and "
                f"cite every factual claim.\n\nGrounding failure: {verdict.detail}"
            )
        if feedback := _reply_script_feedback(ctx.deps.query, output):
            raise ModelRetry(feedback)
        return output

    def conversation(self) -> "_PydanticConversation":
        return _PydanticConversation(self)

    def conversation_from_state(self, state: bytes) -> "_PydanticConversation":
        return _PydanticConversation.from_state(self, state)

    async def run(
        self,
        user_message: str,
        *,
        reminders: list[str] | None = None,
        output_dir: Path | None = None,
        drafts: Any = None,
        resident_facts: dict[str, ResidentFact] | None = None,
        **_: Any,
    ) -> AgentResult:
        return await self.conversation().send(
            user_message,
            reminders=reminders,
            output_dir=output_dir,
            drafts=drafts,
            resident_facts=resident_facts,
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
            return (
                AgentResult(
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
                ),
                new_messages,
                None,
            )
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
        except UsageLimitExceeded:
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
            result = self._project_result(
                new_messages,
                _captured_usage(new_messages),
                EMPTY_ANSWER_FALLBACK,
                citations,
                started,
                model_time_ms=timing_capability.elapsed_ms,
                status="max_turns",
            )
            result.hit_max_iters = True
            return result, new_messages, None
        result = self._result(
            native,
            citations,
            started,
            model_time_ms=timing_capability.elapsed_ms,
        )
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
        output: str | DeferredToolRequests,
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
        pending = isinstance(output, DeferredToolRequests)
        iterations = sum(isinstance(message, ModelResponse) for message in new_messages)
        cost, cost_source = _complete_cost(self.model, new_messages, usage)
        text = (
            ""
            if pending
            else attach_temporal_provenance(
                str(output),
                citations.mapping(),
            )
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
        timing_capability = _ModelTimingCapability()
        native = await self.runtime._agent.run(
            message_history=self._history,
            deferred_tool_results=DeferredToolResults(approvals=approvals),
            deps=deps,
            capabilities=[timing_capability],
        )
        result = self.runtime._result(
            native,
            citations,
            started,
            model_time_ms=timing_capability.elapsed_ms,
        )
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
        state = self.store.pop_pending_approval(self.user_key)
        if state is None:
            raise ValueError("Pending approval expired or already consumed")
        self.conversation = self.runtime.conversation_from_state(state)
        approvals = (
            {
                call_id: decision
                for call_id in self.conversation.pending_approvals
            }
            if isinstance(decision, bool)
            else decision
        )
        result = await self.conversation.resume_approvals(approvals, **kwargs)
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
        requests = tuple(self.conversation.pending_approvals.values())
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
            "Reply YES if these facts are accurate, or NO to correct them.",
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
    )
