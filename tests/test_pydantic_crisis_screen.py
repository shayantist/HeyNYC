from types import SimpleNamespace

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from heynyc.core.crisis_lines import CRISIS_LINES, crisis_response
from heynyc.core.pydantic_runtime import PydanticRunFailure, PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import _reply_language
from heynyc.core.pydantic_runtime.safety import build_crisis_screen
from heynyc.core.registry import Registry


async def test_crisis_screen_uses_typed_output_and_reports_native_usage() -> None:
    async def classify(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        assert "আজ রাতে" in str(messages)
        output = info.output_tools[0]
        return ModelResponse(
            [
                ToolCallPart(
                    output.name,
                    {
                        "risk": "imminent_self_harm",
                        "language": "bn",
                    },
                    "risk-1",
                )
            ],
            usage=RequestUsage(
                input_tokens=20,
                output_tokens=3,
                cache_read_tokens=5,
            ),
        )

    screen = build_crisis_screen(
        FunctionModel(classify),
        model_name="test/safety",
    )

    result = await screen(("আমি আজ রাতে আত্মহত্যা করব",))

    assert result.risk == "imminent_self_harm"
    assert result.language == "bn"
    assert not hasattr(result, "response")
    assert result.model == "test/safety"
    assert result.input_tokens == 20
    assert result.output_tokens == 3
    assert result.cached_input_tokens == 5
    assert result.requests == 1


async def test_semantic_crisis_screen_bypasses_the_answer_model_and_counts_usage() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("This must not run.")])

    async def screen(user_turns: tuple[str, ...]):
        assert user_turns[-1] == "আমি আজ রাতে আত্মহত্যা করব"
        return SimpleNamespace(
            risk="imminent_self_harm",
            language="bn",
            model="test/safety",
            input_tokens=20,
            output_tokens=3,
            cached_input_tokens=5,
            requests=1,
            cost_usd=0.001,
            latency_ms=4.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("আমি আজ রাতে আত্মহত্যা করব")

    assert answer_calls == 0
    assert result.text.startswith(crisis_response("imminent_self_harm", "bn"))
    assert CRISIS_LINES["bn"].lifeline_988 not in result.text
    assert result.usage["safety_model"] == "test/safety"
    assert result.usage["safety_input_tokens"] == 20
    assert result.usage["input_tokens"] == 20
    assert result.usage["n_model_calls"] == 1
    assert result.usage["cost_usd"] == 0.001
    assert result.diagnostics["safety_response_source"] == "deterministic"
    assert result.diagnostics["safety_language"] == "bn"


async def test_semantic_crisis_screen_does_not_capture_third_person_help() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("Grounded third-person guidance.")])

    async def screen(_user_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="none",
            language="en",
            model="test/safety",
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0005,
            latency_ms=2.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("How can I help a friend who says they want to die?")

    assert result.text == "Grounded third-person guidance."
    assert answer_calls == 1
    assert result.usage["n_model_calls"] == 2
    assert result.usage["safety_input_tokens"] == 10


async def test_unavailable_crisis_screen_fails_closed_before_answering() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("Normal service answer.")])

    async def unavailable(_user_turns: tuple[str, ...]):
        raise TimeoutError

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=unavailable,
    )

    result = await runtime.run("Where is the nearest SNAP center?")

    assert "temporary problem" in result.text
    assert "911" in result.text
    assert answer_calls == 0
    assert result.diagnostics["safety_error"] == "TimeoutError"
    assert result.usage["safety_error"] == "TimeoutError"


async def test_crisis_screen_retries_an_unknown_response_language() -> None:
    calls = 0

    async def classify(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        output = info.output_tools[0]
        if calls == 1:
            return ModelResponse([
                ToolCallPart(
                    output.name,
                    {
                        "risk": "imminent_self_harm",
                        "language": "xx",
                    },
                    "risk-1",
                )
            ])
        assert any(
            isinstance(part, RetryPromptPart)
            for message in messages
            for part in message.parts
        )
        return ModelResponse([
            ToolCallPart(
                output.name,
                {
                    "risk": "imminent_self_harm",
                    "language": "bn",
                },
                "risk-2",
            )
        ])

    result = await build_crisis_screen(
        FunctionModel(classify),
        model_name="test/safety",
    )(("আমি আজ রাতে আত্মহত্যা করব",))

    assert calls == 2
    assert result.language == "bn"


async def test_crisis_screen_does_not_accept_model_authored_response_text() -> None:
    async def classify(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        output = info.output_tools[0]
        assert "response" not in output.parameters_json_schema["properties"]
        return ModelResponse([
            ToolCallPart(
                output.name,
                {"risk": "self_harm", "language": "en"},
                "risk-1",
            )
        ])

    result = await build_crisis_screen(
        FunctionModel(classify),
        model_name="test/safety",
    )(("I want to die.",))

    assert result.language == "en"
    assert not hasattr(result, "response")


async def test_runtime_fails_closed_on_a_missing_risk_language() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("This must not run.")])

    async def malformed(_user_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="self_harm",
            language=None,
            model="test/safety",
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0005,
            latency_ms=2.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=malformed,
    )

    result = await runtime.run("Where can I get food?")

    assert "temporary problem" in result.text
    assert "911" in result.text
    assert result.diagnostics["safety_error"] == "MissingCrisisLanguage"
    assert result.usage["safety_error"] == "MissingCrisisLanguage"
    assert answer_calls == 0


async def test_runtime_fails_closed_on_invalid_language_for_no_risk() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("This must not run.")])

    async def malformed(_user_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="none",
            language="xx",
            model="test/safety",
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0005,
            latency_ms=2.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=malformed,
    )

    result = await runtime.run("Where can I get food?")

    assert "911" in result.text
    assert result.diagnostics["safety_error"] == "InvalidCrisisLanguage"
    assert result.usage["safety_error"] == "InvalidCrisisLanguage"
    assert answer_calls == 0


async def test_runtime_fails_closed_on_unhashable_language() -> None:
    answer_calls = 0

    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("This must not run.")])

    async def malformed(_user_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="none",
            language=["en"],
            model="test/safety",
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0005,
            latency_ms=2.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=malformed,
    )

    result = await runtime.run("Where can I get food?")

    assert "temporary problem" in result.text
    assert "911" in result.text
    assert result.diagnostics["safety_error"] == "InvalidCrisisLanguage"
    assert result.usage["safety_error"] == "InvalidCrisisLanguage"
    assert answer_calls == 0


async def test_screened_downstream_failure_does_not_add_911() -> None:
    async def answer(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        raise UnexpectedModelBehavior("answer provider down")

    screen, _ = _screen_returning("none")
    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
    )

    with pytest.raises(PydanticRunFailure) as failed:
        await runtime.run("Where can I renew Medicaid?")

    result = failed.value.partial_result
    assert "temporary problem" in result.text
    assert "311" in result.text
    assert "911" not in result.text
    assert result.diagnostics["safety_language"] == "en"


async def test_runtime_ignores_injected_model_authored_crisis_text() -> None:
    async def screen(_user_turns: tuple[str, ...]):
        return SimpleNamespace(
            risk="imminent_self_harm",
            language="bn",
            response="Take pills.",
            model="test/safety",
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0005,
            latency_ms=2.0,
        )

    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([TextPart("must not run")])),
        registry=Registry([]),
        tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("আমি আজ রাতে আত্মহত্যা করব")

    assert "Take pills" not in result.text
    assert result.text.startswith(crisis_response("imminent_self_harm", "bn"))
    assert result.diagnostics["safety_response_source"] == "deterministic"


async def test_runtime_does_not_classify_resident_text_without_the_screen() -> None:
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([TextPart("must not run")])),
        registry=Registry([]),
        tools={},
    )

    result = await runtime.run("I'm going to kill myself tonight")

    assert result.text == "must not run"
    assert "safety_risk" not in result.diagnostics


# F146: resident-language safety meaning belongs to the typed model screen
def _screen_returning(risk: str, language: str = "en"):
    calls: list[tuple[str, ...]] = []

    async def screen(user_turns: tuple[str, ...]):
        calls.append(user_turns)
        return SimpleNamespace(
            risk=risk, language=language, model="test/safety", input_tokens=1,
            output_tokens=1, cached_input_tokens=0, requests=1, cost_usd=0.0, latency_ms=1.0,
        )

    return screen, calls


def _unreachable_answer():
    async def answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise AssertionError("the deterministic floor must bypass the answer model")

    return answer


async def test_screen_clears_a_therapeutic_dose_false_positive() -> None:
    """The screen reads the whole message instead of matching a dose phrase."""
    answered = False

    async def answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal answered
        answered = True
        return ModelResponse([TextPart("Ordinary answer.")])

    screen, calls = _screen_returning("none")
    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer), registry=Registry([]), tools={}, crisis_screen=screen,
    )

    result = await runtime.run("I just took 2 ibuprofen pills for my headache")

    assert calls
    assert answered
    assert "988" not in result.text
    assert result.diagnostics["safety_risk"] == "none"
    assert "safety_response_source" not in result.diagnostics


async def test_screen_verdict_is_the_only_resident_language_safety_decision() -> None:
    screen, calls = _screen_returning("none")
    answered = False

    async def answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal answered
        answered = True
        return ModelResponse([TextPart("Screen found no self-harm risk.")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer), registry=Registry([]), tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("I'm going to kill myself")

    assert calls
    assert answered
    assert result.text == "Screen found no self-harm risk."
    assert result.diagnostics["safety_risk"] == "none"


async def test_screen_finding_crisis_serves_the_residents_language() -> None:
    screen, _ = _screen_returning("imminent_self_harm", language="ur")
    runtime = PydanticRuntimeAdapter(
        FunctionModel(_unreachable_answer()), registry=Registry([]), tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("میں آج رات خودکشی کرنے کا ارادہ رکھتا ہوں")

    assert CRISIS_LINES["ur"].lifeline_988 in result.text
    assert result.diagnostics["safety_risk"] == "imminent_self_harm"
    assert result.diagnostics["safety_language"] == "ur"


async def test_non_self_harm_medical_question_reaches_the_answer_model() -> None:
    answer_calls = 0

    async def answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal answer_calls
        answer_calls += 1
        return ModelResponse([TextPart("Call 911 right now.")])

    screen, _ = _screen_returning("none")
    runtime = PydanticRuntimeAdapter(
        FunctionModel(answer), registry=Registry([]), tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("I have severe chest pain right now")

    assert "911" in result.text
    assert answer_calls == 1
    assert result.diagnostics["safety_risk"] == "none"


async def test_screen_failure_returns_the_general_emergency_fallback() -> None:

    async def screen(_user_turns: tuple[str, ...]):
        raise RuntimeError("safety provider down")

    runtime = PydanticRuntimeAdapter(
        FunctionModel(_unreachable_answer()), registry=Registry([]), tools={},
        crisis_screen=screen,
    )

    result = await runtime.run("I'm going to kill myself")

    assert "988" not in result.text
    assert "911" in result.text
    assert result.diagnostics["safety_error"] == "RuntimeError"


# F156: per-turn language from the screen, the standard multilingual-chat pattern
# Detection is on the INPUT; the output-side script check could never see Spanish
@pytest.mark.parametrize(
    ("language", "message", "expected"),
    [
        ("en", "is it open on saturday?", "English"),
        ("es", "¿hay alguna despensa de alimentos abierta ahora?", "Spanish"),
        ("bn", "আমার কাছাকাছি ফুড প্যান্ট্রি কোথায়?", "Bengali"),
        # the code that drifted: "Reply in ht" asks the model to decode an abbreviation
        ("ht", "ou te pale de yon red card pi bone, kisa sa ye", "Haitian Creole"),
    ],
)
def test_reply_language_follows_the_current_turn(language, message, expected):
    assert _reply_language(SimpleNamespace(language=language), (message,)) == expected


@pytest.mark.parametrize("message", ["ok gracias", "yes", "ok", "si"])
def test_typed_language_result_is_not_overridden_by_message_length(message):
    assert _reply_language(SimpleNamespace(language="es"), (message,)) == "Spanish"


def test_no_screen_result_means_no_language_instruction():
    assert _reply_language(None, ("where is the nearest food pantry",)) == ""
