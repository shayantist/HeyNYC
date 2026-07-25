from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from litellm.main import responses_api_bridge_check
from pydantic_ai import UsageLimits
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from heynyc.core import config
from heynyc.core.memory import compact_memory, context_capacity, request_tokens
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
        usage_limits=UsageLimits(request_limit=10 if use_module_capabilities else 8),
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
