from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from litellm.main import responses_api_bridge_check
from pydantic_ai import UsageLimits
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from heynyc.core import config
from heynyc.core.memory import compact_memory, context_capacity, request_tokens
from heynyc.core.nli import PromptedNLI
from heynyc.core.prompts import build_system_prompt_tiers
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.base import Tool

from .approvals import PydanticApprovalFlow, _approval_copy, approval_review_text
from .projection import (
    GroundedAnswer,
    GroundedBlock,
    _complete_cost,
    _dynamic_instructions,
    _measurement_messages,
    _native_cache_settings,
    _native_cost,
    _native_orchestration_history,
    _resident_history,
    _semantic_citation_evidence,
)
from .runtime import PydanticRunFailure, PydanticRuntimeAdapter
from .safety import build_crisis_screen, build_output_moderator
from .tools import (
    _resident_fact_errors,
    adapt_tool,
    build_module_capabilities,
    resident_fact_confirmation_tool,
)

__all__ = (
    "PydanticApprovalFlow",
    "PydanticRunFailure",
    "PydanticRuntimeAdapter",
    "GroundedAnswer",
    "GroundedBlock",
    "_approval_copy",
    "_complete_cost",
    "_dynamic_instructions",
    "_measurement_messages",
    "_native_cache_settings",
    "_native_cost",
    "_native_orchestration_history",
    "_resident_fact_errors",
    "_resident_history",
    "_semantic_citation_evidence",
    "adapt_tool",
    "approval_review_text",
    "build_configured_runtime",
    "build_crisis_screen",
    "build_output_moderator",
    "build_module_capabilities",
    "build_runtime",
    "compact_memory",
    "configured_model",
    "context_capacity",
    "resident_fact_confirmation_tool",
    "request_tokens",
    "_uses_openai_responses",
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
    structured_grounding: bool = True,
    semantic_verifier: Any = None,
    fact_review_model: Any = None,
    fact_review_model_name: str = "",
    stream_model_requests: bool = False,
    crisis_screen: Any = None,
    output_guard: Any = None,
) -> PydanticRuntimeAdapter:
    """Build the isolated parity runtime around a caller-selected Pydantic model."""
    runtime_tools = tools if tools is not None else build_toolbox(registry, index=index)
    has_governed_tools = any(tool.resident_fact_scope for tool in runtime_tools.values())
    use_module_capabilities = use_module_capabilities or has_governed_tools
    if has_governed_tools and fact_review_model is None:
        fact_review_model = model
        fact_review_model_name = (
            fact_review_model_name
            or answer_model_route
            or getattr(model, "model_name", type(model).__name__)
        )
    stable_prompt, _ = build_system_prompt_tiers(
        registry,
        query="",
        include_module_guidance=not use_module_capabilities,
    )
    return PydanticRuntimeAdapter(
        model,
        registry=registry,
        tools=runtime_tools,
        system_prompt=stable_prompt,
        prompt_builder=lambda query: build_system_prompt_tiers(
            registry,
            query=query,
            include_module_guidance=not use_module_capabilities,
        )[1],
        use_module_capabilities=use_module_capabilities,
        current_awareness=current_awareness,
        extra_capabilities=extra_capabilities,
        usage_limits=UsageLimits(request_limit=10 if use_module_capabilities else 8),
        answer_model_route=answer_model_route,
        structured_grounding=structured_grounding,
        semantic_verifier=semantic_verifier,
        fact_review_model=fact_review_model,
        fact_review_model_name=fact_review_model_name,
        stream_model_requests=stream_model_requests,
        crisis_screen=crisis_screen,
        output_guard=output_guard,
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


def configured_model(
    model: str,
    *,
    reasoning_effort: str | None = None,
) -> Any:
    reasoning_effort = reasoning_effort or config.HEYNYC_REASONING_EFFORT
    if model.startswith("openai/"):
        settings = {
            key: value
            for key, value in {
                "openai_reasoning_effort": reasoning_effort,
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
    model: Any,
    index: Any = None,
    current_awareness: Callable[[], Awaitable[str]] | None = None,
    output_guard: Any = None,
) -> PydanticRuntimeAdapter:
    selected_model = configured_model(model) if isinstance(model, str) else model
    safety_model_name = (
        config.HEYNYC_SCOPE_MODEL
        if isinstance(model, str)
        else type(selected_model).__name__
    )
    safety_model = (
        configured_model(safety_model_name, reasoning_effort="low")
        if isinstance(model, str)
        else selected_model
    )
    return build_runtime(
        registry,
        model=selected_model,
        tools=build_toolbox(registry, index=index),
        use_module_capabilities=True,
        current_awareness=current_awareness,
        answer_model_route=model if isinstance(model, str) else None,
        structured_grounding=True,
        semantic_verifier=(
            PromptedNLI(config.HEYNYC_CITATION_CHECK_MODEL)
            if isinstance(model, str)
            else None
        ),
        fact_review_model=(
            configured_model(config.HEYNYC_FACT_REVIEW_MODEL)
            if isinstance(model, str)
            else selected_model
        ),
        fact_review_model_name=(
            config.HEYNYC_FACT_REVIEW_MODEL
            if isinstance(model, str)
            else type(selected_model).__name__
        ),
        stream_model_requests=True,
        crisis_screen=build_crisis_screen(
            safety_model,
            model_name=safety_model_name,
        ),
        output_guard=output_guard,
    )
