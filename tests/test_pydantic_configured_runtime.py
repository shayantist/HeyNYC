from pydantic_ai.models.test import TestModel

from heynyc.core.pydantic_runtime import build_configured_runtime, configured_model
from heynyc.core.registry import Registry


def test_configured_runtime_uses_structured_grounding_without_uncalibrated_semantic_filter(
    monkeypatch,
):
    captured = {}

    def build_runtime(_registry, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_runtime",
        build_runtime,
    )
    model = TestModel()
    build_configured_runtime(Registry([]), model=model)

    assert captured["structured_grounding"] is True
    assert captured["use_module_capabilities"] is True
    assert captured.get("semantic_verifier") is None
    assert captured["model"] is model
    assert captured["fact_review_model"] is model


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
