import asyncio
import json

import pytest
from pydantic_ai import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from heynyc.core.agent import AgentResult
from heynyc.core.events import (
    Done,
    ModelRequestCompleted,
    ModelRequestStart,
    OutputAttempt,
    ToolCompleted,
    ToolStart,
    ValidationRejected,
    to_sse,
)
from heynyc.core.pydantic_runtime import build_configured_runtime
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.eval.cases import EvalCase
from heynyc.eval.report import event_writer
from heynyc.eval.runner import run_case


async def test_run_case_forwards_runtime_events() -> None:
    seen = []

    class Agent:
        async def run(self, query, reminders=None, event_sink=None):
            event_sink(ToolStart("call-1", "web_search", "Searching"))
            return AgentResult(text="answer", citations={}, tool_calls_made=["web_search"])

    result = await run_case(
        Agent(),
        EvalCase(id="live-events", module="global", query="What happened?"),
        event_sink=seen.append,
    )

    assert result.text == "answer"
    assert [event.name for event in seen if isinstance(event, ToolStart)] == ["web_search"]


def test_event_writer_flushes_each_runtime_event(tmp_path) -> None:
    write = event_writer(tmp_path, "live-events")

    write(ToolStart("call-1", "web_search", "Searching"))
    write(Done("success", 2))

    rows = [json.loads(line) for line in (tmp_path / "events" / "live-events.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["tool.start", "done"]
    assert rows[0]["name"] == "web_search"


def test_event_writer_keeps_full_tool_payloads_out_of_resident_sse(tmp_path) -> None:
    started = ToolStart(
        "call-1",
        "web_fetch",
        "Fetching",
        args={"url": "https://example.com", "query": "private eval query"},
    )
    completed = ToolCompleted(
        "call-1",
        "web_fetch",
        "ok",
        "short preview",
        result="complete fetched evidence",
    )

    assert "private eval query" not in to_sse(started)
    assert "complete fetched evidence" not in to_sse(completed)

    write = event_writer(tmp_path, "tool-payloads")
    write(started)
    write(completed)
    rows = [
        json.loads(line)
        for line in (tmp_path / "events" / "tool-payloads.jsonl").read_text().splitlines()
    ]

    assert rows[0]["args"] == {
        "url": "https://example.com",
        "query": "private eval query",
    }
    assert rows[1]["result"] == "complete fetched evidence"


async def test_configured_runtime_forwards_exact_tool_payloads(monkeypatch) -> None:
    async def lookup(args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://example.com/library",
            title="Library",
            kind="WEB",
            snippet="Open until 8 PM",
            provenance={"evidence_grade": "authoritative"},
        )
        return f"Open until 8 PM for {args['branch']} {{cite:{citation_id}}}"

    tool = Tool(
        name="lookup_library",
        description="Look up one library branch",
        parameters={
            "type": "object",
            "properties": {"branch": {"type": "string"}},
            "required": ["branch"],
        },
        handler=lookup,
    )
    calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {
                0: DeltaToolCall(
                    name="lookup_library",
                    json_args=json.dumps({"branch": "Sunset Park"}),
                    tool_call_id="lookup-1",
                )
            }
            return
        final = next(tool for tool in info.output_tools if tool.name == "final_answer")
        citation_id = "S9" if calls == 2 else "S1"
        yield {
            0: DeltaToolCall(
                name=final.name,
                json_args=json.dumps(
                    {
                        "answer": (
                            f"Sunset Park is open until 8 PM. {{cite:{citation_id}}}"
                        ),
                    },
                ),
                tool_call_id=f"final-{calls}",
            )
        }

    async def no_screen(_turns):
        return None

    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_toolbox",
        lambda _registry, index=None: {tool.name: tool},
    )
    monkeypatch.setattr(
        "heynyc.core.pydantic_runtime.build_crisis_screen",
        lambda _model, *, model_name: no_screen,
    )
    seen = []
    runtime = build_configured_runtime(
        Registry([]),
        model=FunctionModel(stream_function=model),
        stream_model_requests=True,
    )

    result = await runtime.run("When does Sunset Park close?", event_sink=seen.append)

    started = next(event for event in seen if isinstance(event, ToolStart))
    completed = next(event for event in seen if isinstance(event, ToolCompleted))
    model_starts = [event for event in seen if isinstance(event, ModelRequestStart)]
    model_completions = [
        event for event in seen if isinstance(event, ModelRequestCompleted)
    ]
    output_attempt = next(event for event in seen if isinstance(event, OutputAttempt))
    rejection = next(event for event in seen if isinstance(event, ValidationRejected))
    assert result.text == "Sunset Park is open until 8 PM. {cite:S1}"
    assert started.args == {"branch": "Sunset Park"}
    assert completed.result == "Open until 8 PM for Sunset Park {cite:S1}"
    assert [event.request_number for event in model_starts] == [1, 2, 3]
    assert [event.request_number for event in model_completions] == [1, 2, 3]
    assert all(event.elapsed_ms >= 0 for event in model_completions)
    assert output_attempt.name == "final_answer"
    assert output_attempt.args == {
        "answer": "Sunset Park is open until 8 PM. {cite:S9}",
    }
    assert rejection.name == "final_answer"
    assert "Use only citation IDs" in rejection.message


async def test_cancelled_eval_keeps_flushed_tool_events(tmp_path) -> None:
    class Agent:
        async def run(self, query, reminders=None, event_sink=None):
            event_sink(
                ToolStart(
                    "call-1",
                    "web_search",
                    "Searching",
                    args={"query": query},
                )
            )
            raise asyncio.CancelledError

    write = event_writer(tmp_path, "cancelled")
    with pytest.raises(asyncio.CancelledError):
        await run_case(
            Agent(),
            EvalCase(id="cancelled", module="global", query="NYC today"),
            event_sink=write,
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "events" / "cancelled.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "timestamp": rows[0]["timestamp"],
            "type": "tool.start",
            "tool_call_id": "call-1",
            "name": "web_search",
            "label": "Searching",
            "args": {"query": "NYC today"},
        }
    ]
