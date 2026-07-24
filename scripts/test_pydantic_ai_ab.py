import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai.models.openai import OpenAIResponsesModel

from heynyc.core.agent import AgentResult
from scripts import pydantic_ai_ab
from scripts.pydantic_ai_ab import build_factories, run_arms, summarize_arm


def test_comparison_model_matches_litellm_responses_bridge(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(pydantic_ai_ab.config, "HEYNYC_REASONING_EFFORT", "medium")

    model = pydantic_ai_ab._comparison_model("openai/gpt-5.4-mini")

    assert isinstance(model, OpenAIResponsesModel)
    assert model.settings["openai_reasoning_effort"] == "medium"
    assert (
        pydantic_ai_ab._runtime_route("pydantic_ai", "openai/gpt-5.4-mini")
        == "pydantic-ai:openai-responses:gpt-5.4-mini"
    )
    assert (
        pydantic_ai_ab._runtime_route("production", "openai/gpt-5.4-mini")
        == "litellm:openai-responses-bridge:openai/gpt-5.4-mini"
    )


def test_responses_bridge_requires_tools(monkeypatch) -> None:
    monkeypatch.setattr(pydantic_ai_ab.config, "HEYNYC_REASONING_EFFORT", "medium")

    assert not pydantic_ai_ab._uses_openai_responses(
        "openai/gpt-5.4-mini",
        has_tools=False,
    )


def test_build_factories_gives_both_arms_the_same_live_awareness(
    monkeypatch,
) -> None:
    built = []

    def fake_agent(registry, **kwargs):
        built.append(("production", registry, kwargs))
        return "production"

    def fake_runtime(registry, **kwargs):
        built.append(("pydantic_ai", registry, kwargs))
        return "pydantic_ai"

    monkeypatch.setattr(pydantic_ai_ab, "Agent", fake_agent)
    monkeypatch.setattr(pydantic_ai_ab, "build_runtime", fake_runtime)
    monkeypatch.setattr(
        pydantic_ai_ab,
        "_comparison_model",
        lambda model: f"chat:{model}",
    )

    factories = build_factories("registry", "retriever", "openai/gpt-test")

    built_factories = {arm: factory() for arm, factory in factories.items()}

    assert built_factories["production"] == "production"
    assert built_factories["pydantic_ai"].runtime == "pydantic_ai"
    assert built == [
        (
            "production",
            "registry",
            {
                "model": "openai/gpt-test",
                "index": "retriever",
                "notify_awareness": pydantic_ai_ab.current_awareness,
                "scope_gate": True,
            },
        ),
        (
            "pydantic_ai",
            "registry",
            {
                "model": "chat:openai/gpt-test",
                "answer_model_route": "openai/gpt-test",
                "index": "retriever",
                "use_module_capabilities": True,
                "current_awareness": pydantic_ai_ab.current_awareness,
            },
        ),
    ]


def test_summarize_arm_counts_every_turn_once_and_marks_unpriced() -> None:
    results = [
        SimpleNamespace(
            case=SimpleNamespace(id="multi"),
            error=None,
            turn_results=[
                SimpleNamespace(
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": 3,
                        "n_model_calls": 1,
                        "requests": 1,
                        "n_tool_calls": 2,
                        "capabilities_used": ["housing"],
                        "latency_ms": 4,
                        "cost_usd": 0.01,
                    }
                ),
                SimpleNamespace(
                    usage={
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "cached_input_tokens": 7,
                        "n_model_calls": 2,
                        "requests": 2,
                        "n_tool_calls": 1,
                        "capabilities_used": ["benefits", "housing"],
                        "latency_ms": 6,
                        "cost_status": "unpriced",
                    }
                ),
            ],
        )
    ]
    report = SimpleNamespace(passed_count=1, total=1)

    assert summarize_arm("pydantic_ai", "openai/gpt-test", results, report) == {
        "arm": "pydantic_ai",
        "runtime": "PydanticRuntimeAdapter",
        "answer_model": "openai/gpt-test",
        "runtime_route": "pydantic-ai:openai-chat:gpt-test",
        "case_ids": ["multi"],
        "passed": 1,
        "total": 1,
        "input_tokens": 30,
        "output_tokens": 7,
        "cached_input_tokens": 10,
        "request_count": 3,
        "model_call_count": 3,
        "tool_call_count": 3,
        "capability_ids": ["benefits", "housing"],
        "latency_ms": 10.0,
        "cost_usd": None,
        "cost_status": "unpriced",
        "error_count": 0,
        "fact_confirmation_policy": "auto_confirm_confirm_star_facts_only_never_actions",
    }


async def test_eval_conversation_merges_native_fact_confirmation_run() -> None:
    pending = AgentResult(
        text="Review facts",
        citations={
            "S0": {"url": "https://nyc.gov/pending-only"},
            "S1": {"url": "https://nyc.gov/food"},
        },
        tool_calls_made=["nearest_food_pantry", "confirm_screen_facts"],
        status="approval_required",
        messages=[{"role": "assistant", "content": "pending"}],
        usage={
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_input_tokens": 3,
            "requests": 1,
            "cost_usd": 0.01,
            "capabilities_used": ["food"],
        },
    )
    final = AgentResult(
        text="Screened",
        citations={
            "S1": {"url": "https://nyc.gov/food"},
            "S2": {"url": "https://access.nyc.gov"},
        },
        tool_calls_made=[],
        messages=[{"role": "tool", "content": "screened"}],
        usage={
            "input_tokens": 20,
            "output_tokens": 4,
            "cached_input_tokens": 5,
            "requests": 2,
            "cost_usd": 0.02,
            "capabilities_used": ["benefits"],
            "executed_tool_calls": ["confirm_screen_facts"],
        },
    )

    class Conversation:
        pending_approvals = {
            "facts-call": {
                "tool_name": "confirm_screen_facts",
                "args": {"profile": {"age": 35}},
            }
        }

        async def send(self, message, **kwargs):
            return pending

        async def resume_approvals(self, decisions):
            assert decisions == {"facts-call": True}
            return final

    result = await pydantic_ai_ab._PydanticEvalConversation(
        Conversation()
    ).send("screen me")

    assert result.text == "Screened"
    assert result.tool_calls_made == [
        "nearest_food_pantry",
        "confirm_screen_facts",
        "screen",
    ]
    assert result.citations == {
        "S0": {"url": "https://nyc.gov/pending-only"},
        "S1": {"url": "https://nyc.gov/food"},
        "S2": {"url": "https://access.nyc.gov"},
    }
    assert result.messages == [
        {"role": "assistant", "content": "pending"},
        {"role": "tool", "content": "screened"},
    ]
    assert result.usage["input_tokens"] == 30
    assert result.usage["output_tokens"] == 6
    assert result.usage["cached_input_tokens"] == 8
    assert result.usage["requests"] == 3
    assert result.usage["cost_usd"] == pytest.approx(0.03)
    assert result.usage["capabilities_used"] == ["food", "benefits"]


def test_merge_does_not_alias_unexecuted_fact_confirmation() -> None:
    pending = AgentResult(
        text="",
        citations={},
        tool_calls_made=["confirm_screen_facts"],
        status="approval_required",
    )
    final = AgentResult(
        text="Could not run the check",
        citations={},
        tool_calls_made=[],
        usage={"executed_tool_calls": []},
    )

    result = pydantic_ai_ab._merge_results(pending, final)

    assert result.tool_calls_made == ["confirm_screen_facts"]


async def test_eval_conversation_never_auto_approves_an_action() -> None:
    pending = AgentResult(
        text="Review action",
        citations={},
        status="approval_required",
    )

    class Conversation:
        pending_approvals = {
            "action-call": {
                "tool_name": "prepare_application",
                "args": {"draft_id": "draft-123"},
            }
        }

        async def send(self, message, **kwargs):
            return pending

        async def resume_approvals(self, decisions):
            raise AssertionError("action approval must stay resident-gated")

    result = await pydantic_ai_ab._PydanticEvalConversation(
        Conversation()
    ).send("prepare it")

    assert result is pending


async def test_run_arms_can_execute_in_parallel(tmp_path, monkeypatch) -> None:
    both_started = asyncio.Event()
    active = 0
    peak = 0
    result = SimpleNamespace(
        case=SimpleNamespace(id="one"),
        error=None,
        turn_results=[SimpleNamespace(usage={})],
    )
    report = SimpleNamespace(passed_count=1, total=1)

    async def fake_run_all(factory, cases, reminders):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        active -= 1
        return [result]

    monkeypatch.setattr(pydantic_ai_ab, "run_all", fake_run_all)
    async def fake_evaluate(results):
        return report

    monkeypatch.setattr(pydantic_ai_ab, "evaluate", fake_evaluate)
    monkeypatch.setattr(pydantic_ai_ab, "write_run", lambda *args, **kwargs: None)

    receipt = await run_arms(
        {"production": lambda: "prod", "pydantic_ai": lambda: "candidate"},
        [SimpleNamespace(id="one")],
        [],
        tmp_path,
        "openai/gpt-test",
        parallel=True,
    )

    assert peak == 2
    assert receipt["parallel"] is True
    assert receipt["performance_comparison_valid"] is False


def test_run_arms_uses_existing_runner_grader_and_writer(
    tmp_path, monkeypatch
) -> None:
    calls = []
    result = SimpleNamespace(
        case=SimpleNamespace(id="one"),
        error=None,
        turn_results=[SimpleNamespace(usage={})],
    )
    report = SimpleNamespace(passed_count=1, total=1)

    async def fake_run_all(factory, cases, reminders):
        calls.append(("run_all", factory(), cases, reminders))
        return [result]

    async def fake_evaluate(results):
        calls.append(("evaluate", results))
        return report

    def fake_write_run(path, written_report, metadata):
        calls.append(("write_run", path, written_report, metadata))

    monkeypatch.setattr(pydantic_ai_ab, "run_all", fake_run_all)
    monkeypatch.setattr(pydantic_ai_ab, "evaluate", fake_evaluate)
    monkeypatch.setattr(pydantic_ai_ab, "write_run", fake_write_run)

    receipt = asyncio.run(
        run_arms(
            {"production": lambda: "prod", "pydantic_ai": lambda: "candidate"},
            [SimpleNamespace(id="one")],
            ["remember"],
            tmp_path,
            "openai/gpt-test",
        )
    )

    assert [call[0] for call in calls] == [
        "run_all",
        "evaluate",
        "write_run",
        "run_all",
        "evaluate",
        "write_run",
    ]
    assert receipt["case_ids"] == ["one"]
    assert [arm["arm"] for arm in receipt["arms"]] == [
        "production",
        "pydantic_ai",
    ]


def test_run_arms_persists_every_turn_and_honors_counterbalanced_order(
    tmp_path, monkeypatch
) -> None:
    calls = []
    case = SimpleNamespace(id="multi", turns=["first", "second"])
    first = AgentResult(
        text="first answer",
        citations={"S1": {"title": "First", "url": "https://nyc.gov/first"}},
        tool_calls_made=["first_tool"],
        messages=[{"role": "assistant", "content": "first answer"}],
        usage={"input_tokens": 10},
    )
    second = AgentResult(
        text="second answer",
        citations={"S2": {"title": "Second", "url": "https://nyc.gov/second"}},
        tool_calls_made=["second_tool"],
        messages=[{"role": "assistant", "content": "second answer"}],
        usage={"input_tokens": 20},
    )
    result = SimpleNamespace(
        case=case,
        error=None,
        turn_results=[first, second],
        turn_started_at=["2026-07-24T10:00:00", "2026-07-24T10:00:01"],
    )
    report = SimpleNamespace(passed_count=1, total=1)

    async def fake_run_all(factory, cases, reminders):
        calls.append(factory())
        return [result]

    async def fake_evaluate(results):
        return report

    monkeypatch.setattr(pydantic_ai_ab, "run_all", fake_run_all)
    monkeypatch.setattr(pydantic_ai_ab, "evaluate", fake_evaluate)
    monkeypatch.setattr(pydantic_ai_ab, "write_run", lambda *args, **kwargs: None)

    receipt = asyncio.run(
        run_arms(
            {"production": lambda: "prod", "pydantic_ai": lambda: "candidate"},
            [case],
            [],
            tmp_path,
            "openai/gpt-test",
            arm_order=("pydantic_ai", "production"),
        )
    )

    assert calls == ["candidate", "prod"]
    assert receipt["arm_order"] == ["pydantic_ai", "production"]
    artifact = pydantic_ai_ab.json.loads(
        (tmp_path / "pydantic_ai" / "turns.json").read_text()
    )
    assert artifact["cases"][0]["turns"] == [
        {
            "turn": 1,
            "started_at": "2026-07-24T10:00:00",
            "resident_message": "first",
            "text": "first answer",
            "status": "success",
            "tool_calls": ["first_tool"],
            "citations": {
                "S1": {"title": "First", "url": "https://nyc.gov/first"}
            },
            "messages": [{"role": "assistant", "content": "first answer"}],
            "usage": {"input_tokens": 10},
        },
        {
            "turn": 2,
            "started_at": "2026-07-24T10:00:01",
            "resident_message": "second",
            "text": "second answer",
            "status": "success",
            "tool_calls": ["second_tool"],
            "citations": {
                "S2": {"title": "Second", "url": "https://nyc.gov/second"}
            },
            "messages": [{"role": "assistant", "content": "second answer"}],
            "usage": {"input_tokens": 20},
        },
    ]
