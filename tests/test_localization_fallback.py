from types import SimpleNamespace

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel

from heynyc.core import localization
from heynyc.core.pydantic_runtime import PydanticRunFailure, PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import (
    TEMPORARY_FAILURE_FALLBACK,
    VERIFICATION_ABSTAIN_FALLBACK,
)
from heynyc.core.registry import Registry


def test_chinese_current_turn_localizes_deterministic_failure_copies() -> None:
    assert localization.localize(TEMPORARY_FAILURE_FALLBACK, "zh-CN") == (
        "我无法完成这次请求，而且本次尝试没有返回来源或部分结果。请重试。"
    )
    assert localization.localize(VERIFICATION_ABSTAIN_FALLBACK, "zh-CN") == (
        "我无法根据找到的可靠来源核实这一点，所以不想猜测。请尝试提供更多细节，我会再帮您查询。"
    )


def test_spanish_current_turn_localizes_verification_abstention() -> None:
    assert localization.localize(VERIFICATION_ABSTAIN_FALLBACK, "es") == (
        "No pude verificarlo con las fuentes confiables que encontré, así que no quiero "
        "adivinar. Intenta preguntar con un poco más de detalle y lo comprobaré de nuevo."
    )


def test_failure_copy_preserves_english_for_non_localized_turns(monkeypatch, tmp_path) -> None:
    assert localization.localize(TEMPORARY_FAILURE_FALLBACK, "en") == (
        TEMPORARY_FAILURE_FALLBACK
    )
    assert localization.localize(TEMPORARY_FAILURE_FALLBACK, "unknown") == (
        TEMPORARY_FAILURE_FALLBACK
    )
    monkeypatch.setattr(localization, "_LOCALE_DIR", tmp_path)
    assert localization.localize(TEMPORARY_FAILURE_FALLBACK, "zh") == (
        TEMPORARY_FAILURE_FALLBACK
    )


def test_partial_chinese_catalog_does_not_add_an_english_welcome_footer() -> None:
    assert localization.welcome_footer(("cooling centers",), "zh") is None


@pytest.mark.asyncio
async def test_pydantic_failure_uses_the_current_screen_language(monkeypatch) -> None:
    async def crisis_screen(_user_turns):
        return SimpleNamespace(
            risk="none",
            language="zh",
            model="test/safety",
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            requests=0,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda: "unused"),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
        crisis_screen=crisis_screen,
    )

    async def fail(*_args, **_kwargs):
        raise UnexpectedModelBehavior("broken output")

    monkeypatch.setattr(runtime._agent, "run", fail)
    with pytest.raises(PydanticRunFailure) as caught:
        await runtime.run("中文地点查询")

    assert caught.value.partial_result.text == (
        "我无法完成这次请求，而且本次尝试没有返回来源或部分结果。请重试。"
    )
