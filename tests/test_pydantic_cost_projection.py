from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage

from heynyc.core import events
from heynyc.core.pydantic_runtime import projection, runtime


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


async def test_live_request_event_uses_the_same_current_pricer(monkeypatch):
    observed = []
    monkeypatch.setattr(runtime, "priced_cost_usd", lambda *_args: 0.0001042)
    capability = runtime._ModelTimingCapability(
        model="openai/gpt-5.6-luna",
        request_timeout_s=5,
    )
    capability.bind(observed.append)

    async def handler(_request_context):
        return ModelResponse(
            parts=[],
            usage=RequestUsage(
                input_tokens=275,
                output_tokens=41,
                cache_read_tokens=10,
            ),
        )

    await capability.wrap_model_request(None, request_context=None, handler=handler)

    completed = next(
        event for event in observed
        if isinstance(event, events.ModelRequestCompleted)
    )
    assert completed.usage["cost_usd"] == 0.0001042
