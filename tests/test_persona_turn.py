import json
from types import SimpleNamespace

import pytest

from heynyc.core.events import Done
from scripts.persona_turn import _run, _trace_result, main


def test_persona_trace_keeps_observable_agent_boundaries() -> None:
    result = SimpleNamespace(
        text="Grounded answer {cite:S1}",
        status="success",
        citations={"S1": {"kind": "DATA", "snippet": "official row"}},
        tool_calls_made=["nearest_snap_center"],
        iterations=2,
        messages=[
            {"role": "user", "content": "where can I apply in person?"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "nearest_snap_center"}}]},
            {"role": "tool", "content": "official row"},
        ],
        usage={"requests": 2, "cost_usd": 0.01},
        diagnostics={"validation_rejections": [{"stage": "discovery_only"}]},
    )

    trace = _trace_result(result)

    assert trace["messages"] == result.messages
    assert trace["citations"] == result.citations
    assert trace["diagnostics"] == result.diagnostics
    assert trace["usage"] == result.usage


def test_persona_cli_loads_project_dotenv_before_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    loaded: list = []

    monkeypatch.setattr(
        "scripts.persona_turn.load_dotenv",
        lambda path: loaded.append(path),
        raising=False,
    )

    def run(coro):
        coro.close()
        assert loaded
        return {"reply": "ok"}

    monkeypatch.setattr("scripts.persona_turn.asyncio.run", run)

    assert main([
        "--user",
        "persona",
        "--data-dir",
        str(tmp_path),
        "help",
    ]) == 0
    assert loaded[0].name == ".env"


@pytest.mark.asyncio
async def test_persona_turn_saves_partial_trace_when_runtime_fails(
    monkeypatch,
    tmp_path,
) -> None:
    partial = SimpleNamespace(
        text="",
        status="error",
        citations={},
        tool_calls_made=["official_sources"],
        iterations=1,
        messages=[{"role": "user", "content": "help"}],
        usage={"requests": 1},
        diagnostics={"validation_rejections": []},
    )
    deps = SimpleNamespace(event_sink=None)

    async def fail(_inbound, _replier, observed_deps) -> None:
        observed_deps.event_sink(Done("error", 1, result=partial))
        raise RuntimeError("provider failed")

    monkeypatch.setattr("heynyc.channels.console.build_console_deps", lambda **_kwargs: deps)
    monkeypatch.setattr("heynyc.channels.orchestrator.handle", fail)

    output = await _run("persona", "help", None, tmp_path)

    assert output["error"]["type"] == "RuntimeError"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["index_enabled"] is False
    assert saved["agent_result"]["messages"] == partial.messages
