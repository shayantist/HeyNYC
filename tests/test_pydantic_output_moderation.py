from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core import events
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.safety import build_output_moderator
from heynyc.core.registry import Registry


class _Categories:
    def __init__(self, values: dict[str, bool]) -> None:
        self.values = values

    def model_dump(self, *, by_alias: bool) -> dict[str, bool]:
        assert by_alias is True
        return self.values


class _Moderations:
    def __init__(self, categories: dict[str, bool], *, flagged: bool) -> None:
        self.categories = categories
        self.flagged = flagged
        self.calls: list[dict[str, str]] = []

    async def create(self, **kwargs: str) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    categories=_Categories(self.categories),
                    flagged=self.flagged,
                )
            ]
        )


@pytest.mark.asyncio
async def test_output_moderator_returns_every_flagged_category() -> None:
    moderations = _Moderations({"self-harm": True}, flagged=True)
    moderate = build_output_moderator(SimpleNamespace(moderations=moderations))

    blocked = await moderate(
        "If you may hurt yourself, call or text 988 for immediate support."
    )

    assert blocked == frozenset({"self-harm"})
    assert moderations.calls == [
        {
            "model": "omni-moderation-latest",
            "input": "If you may hurt yourself, call or text 988 for immediate support.",
        }
    ]


@pytest.mark.asyncio
async def test_output_moderator_returns_blocking_categories() -> None:
    moderations = _Moderations(
        {"self-harm/instructions": True, "violence": True},
        flagged=True,
    )
    moderate = build_output_moderator(SimpleNamespace(moderations=moderations))

    assert await moderate("unsafe") == frozenset({
        "self-harm/instructions",
        "violence",
    })


@pytest.mark.asyncio
async def test_injected_output_guard_blocks_without_a_model_retry() -> None:
    calls = 0
    reviewed: list[str] = []

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ):
        nonlocal calls
        calls += 1
        yield "unsafe instructions"

    async def output_guard(text: str) -> frozenset[str]:
        reviewed.append(text)
        return frozenset({"self-harm/instructions"})

    emitted: list[events.Event] = []
    runtime = PydanticRuntimeAdapter(
        FunctionModel(stream_function=model),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
        output_guard=output_guard,
    )

    result = await runtime.run("Help", event_sink=emitted.append)

    assert calls == 1
    assert result.status == "error"
    assert "unsafe instructions" not in result.text
    assert reviewed == ["unsafe instructions"]
    assert not [event for event in emitted if isinstance(event, events.TextDelta)]
    completed = [
        event for event in emitted if isinstance(event, events.MessageCompleted)
    ]
    assert [event.text for event in completed] == [result.text]
    errors = [event for event in emitted if isinstance(event, events.ErrorEvent)]
    assert [event.retryable for event in errors] == [False]


@pytest.mark.asyncio
async def test_output_guard_is_off_by_default() -> None:
    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ordinary answer")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
    )

    result = await runtime.run("Help")

    assert result.text == "ordinary answer"


@pytest.mark.asyncio
async def test_output_guard_failure_fails_closed_without_a_model_retry() -> None:
    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ):
        nonlocal calls
        calls += 1
        yield "unreviewed answer"

    async def unavailable(_text: str) -> frozenset[str]:
        raise RuntimeError("offline")

    emitted: list[events.Event] = []
    runtime = PydanticRuntimeAdapter(
        FunctionModel(stream_function=model),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
        output_guard=unavailable,
    )

    result = await runtime.run("Help", event_sink=emitted.append)

    assert calls == 1
    assert result.status == "error"
    assert "unreviewed answer" not in result.text
    assert not [event for event in emitted if isinstance(event, events.TextDelta)]
    errors = [event for event in emitted if isinstance(event, events.ErrorEvent)]
    assert [event.retryable for event in errors] == [False]
