from types import SimpleNamespace

from heynyc.core.pydantic_runtime import projection


def test_complete_cost_prefers_current_repo_pricing_for_a_known_model(monkeypatch):
    monkeypatch.setattr(projection, "_native_cost", lambda _messages: 0.000521)
    monkeypatch.setattr(projection, "priced_cost_usd", lambda *_args: 0.0001042)

    cost, source = projection._complete_cost(
        "openai/gpt-5.6-luna",
        [],
        SimpleNamespace(
            input_tokens=275,
            output_tokens=41,
            cache_read_tokens=0,
        ),
    )

    assert cost == 0.0001042
    assert source == "litellm-fallback"
