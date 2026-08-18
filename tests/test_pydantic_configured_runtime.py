import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from heynyc.core import config
from heynyc.core.pydantic_runtime import (
    build_configured_runtime,
    build_runtime,
    configured_model,
)
from heynyc.core.registry import Registry
from heynyc.core.tools import build_toolbox
from heynyc.core.tools.base import Tool, ToolContext


def test_configured_runtime_uses_structured_grounding_without_uncalibrated_semantic_filter(
    monkeypatch,
):
    captured = {}
    safety_screen = object()
    scope_screen = object()
    output_guard = object()

    def build_runtime(_registry, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_runtime",
        build_runtime,
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_crisis_screen",
        lambda _model, *, model_name: (
            safety_screen
            if model_name == "TestModel"
            else AssertionError(model_name)
        ),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_scope_screen",
        lambda _model, *, model_name, registry: (
            scope_screen if model_name == "TestModel" else AssertionError(model_name)
        ),
    )
    model = TestModel()
    build_configured_runtime(
        Registry([]),
        model=model,
        output_guard=output_guard,
    )

    assert captured["structured_grounding"] is True
    assert captured["use_module_capabilities"] is True
    assert captured.get("semantic_verifier") is None
    assert captured["model"] is model
    assert captured["fact_review_model"] is model
    assert captured["crisis_screen"] is safety_screen
    assert captured["scope_screen"] is scope_screen
    assert captured["output_guard"] is output_guard


def test_configured_runtime_keeps_uncalibrated_output_moderation_off(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_runtime",
        lambda _registry, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_crisis_screen",
        lambda _model, *, model_name: object(),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_scope_screen",
        lambda _model, *, model_name, registry: object(),
    )

    build_configured_runtime(Registry([]), model=TestModel())

    assert captured["output_guard"] is None


async def test_configured_runtime_preserves_typed_provider_failure_result() -> None:
    class ProviderResult(BaseModel):
        records: list[dict]
        next_cursor: str | None
        error: str | None

    async def handler(_args: dict, _ctx: ToolContext) -> dict:
        return {
            "records": [],
            "next_cursor": None,
            "error": "provider unavailable",
        }

    seen: list[object] = []

    async def model(messages, _info) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse([ToolCallPart("provider_lookup", {}, "provider-1")])
        seen.append(returns[-1].content)
        return ModelResponse([TextPart("Done")])

    runtime = build_runtime(
        Registry([]),
        model=FunctionModel(model),
        tools={
            "provider_lookup": Tool(
                name="provider_lookup",
                description="Return one provider result",
                parameters={"type": "object", "properties": {}},
                handler=handler,
                return_type=ProviderResult,
            )
        },
        structured_grounding=False,
    )

    result = await runtime.run("Find it")

    assert result.status == "success"
    assert len(seen) == 1
    assert isinstance(seen[0], ProviderResult)
    assert seen[0].model_dump() == {
        "records": [],
        "next_cursor": None,
        "error": "provider unavailable",
    }


def test_configured_structured_runtime_does_not_stream_model_requests(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_runtime",
        lambda _registry, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_crisis_screen",
        lambda _model, *, model_name: object(),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_scope_screen",
        lambda _model, *, model_name, registry: object(),
    )

    build_configured_runtime(Registry([]), model=TestModel())

    assert captured["stream_model_requests"] is False


def test_configured_runtime_uses_configured_semantic_checker_in_public_path(
    monkeypatch,
):
    captured = {}
    configured = []
    verifier_models = []
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.configured_model",
        lambda model, **_kwargs: configured.append(model) or object(),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_crisis_screen",
        lambda _model, *, model_name: object(),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_scope_screen",
        lambda _model, *, model_name, registry: object(),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_runtime",
        lambda _registry, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.PromptedNLI",
        lambda model: verifier_models.append(model) or object(),
        raising=False,
    )

    build_configured_runtime(Registry([]), model="openai/gpt-5.6-luna")

    assert captured["semantic_verifier"] is not None
    assert verifier_models == [config.HEYNYC_CITATION_CHECK_MODEL]
    assert "openai/gpt-5.6-luna" in configured


def test_configured_model_delegates_non_openai_providers_to_pydantic(
    monkeypatch,
) -> None:
    observed = []
    sentinel = object()
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.infer_model",
        lambda model: observed.append(model) or sentinel,
    )

    assert configured_model("openrouter/qwen/qwen3.6-35b-a3b") is sentinel
    assert observed == ["openrouter:qwen/qwen3.6-35b-a3b"]


def test_configured_model_can_lower_reasoning_for_a_mechanical_classifier(
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime._uses_openai_responses",
        lambda _model: True,
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.OpenAIResponsesModel",
        lambda model, *, settings, profile: captured.update(
            model=model,
            settings=settings,
            profile=profile,
        ),
    )

    configured_model(
        "openai/gpt-5.4-mini",
        reasoning_effort="low",
    )

    assert captured["model"] == "gpt-5.4-mini"
    assert captured["settings"]["openai_reasoning_effort"] == "low"


def test_configured_luna_disables_rejected_native_tool_search(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime._uses_openai_responses",
        lambda _model: True,
    )

    model = configured_model("openai/gpt-5.6-luna")

    native_names = {
        tool.__name__ for tool in model.profile["supported_native_tools"]
    }
    assert "ToolSearchTool" not in native_names


@pytest.mark.parametrize("with_index", [False, True])
async def test_configured_luna_prepared_request_hides_undiscovered_tools(
    monkeypatch,
    with_index,
) -> None:
    captured = {}

    class Index:
        def search(self, _query, k=5):
            return []

    async def capture(_messages, info):
        captured["parameters"] = info.model_request_parameters
        captured["settings"] = info.model_settings
        return ModelResponse([
            ToolCallPart("final_answer", {"answer": "No factual claim."}, "final-1")
        ])

    registry = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    index = Index() if with_index else None
    runtime = build_runtime(
        registry,
        model=FunctionModel(capture),
        tools=build_toolbox(registry, index=index),
        use_module_capabilities=True,
        structured_grounding=True,
    )
    await runtime.run("hello")

    raw_names = {tool.name for tool in captured["parameters"].function_tools}
    assert {"index_search", "search_official_guidance"}.isdisjoint(raw_names)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime._uses_openai_responses",
        lambda _model: True,
    )
    model = configured_model("openai/gpt-5.6-luna", reasoning_effort="low")
    _, prepared = model.prepare_request(
        captured["settings"], captured["parameters"]
    )
    names = {tool.name for tool in prepared.function_tools}

    expected = {
        "load_capability",
        "geocode",
        "nearest",
        "distance",
        "web_search",
        "web_fetch",
        "about_heynyc",
        "search_tools",
    }
    assert names == expected
    search_tools = next(
        tool for tool in prepared.function_tools if tool.name == "search_tools"
    )
    assert "purpose-built NYC service workflows" in search_tools.description
    assert "general web facts" in search_tools.description
    assert {tool.name for tool in prepared.output_tools} == {
        "final_answer",
        "grounded_answer",
        "clarification_request",
        "nonfactual_outcome",
    }
    assert not any(tool.defer_loading for tool in prepared.function_tools)
