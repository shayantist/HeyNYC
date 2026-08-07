from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.citations import CitationRegistry, content_hash
from heynyc.core.localization import localize
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import GeoPoint
from heynyc.modules.cooling_centers import tools as cooling


def _row(space_type: str, location_type: str, *, closed: bool) -> dict:
    return {
        "OBJECTID": 1 if space_type == "Cooling Center" else 2,
        "NYCEM_ID": space_type,
        "Facility_name": f"{space_type} site",
        "Address": "123 W 42 ST",
        "lat": 40.7580,
        "lon": -73.9780,
        "Finder_status": "OPEN",
        "Space_type": space_type,
        "Location_type": location_type,
        "cc_fri_open1": "09:00 AM",
        "cc_fri_close1": "10:00 AM" if closed else "05:00 PM",
    }


def _patch_cooling(monkeypatch, rows: list[dict]) -> None:
    async def geocode(_text: str, **_kwargs):
        return GeoPoint(40.7580, -73.9780, "Times Square")

    async def query(_url: str, **_kwargs):
        return rows

    monkeypatch.setattr(cooling, "geocode", geocode)
    monkeypatch.setattr(cooling, "query_feature_service", query)
    monkeypatch.setattr(
        cooling,
        "_nyc_now",
        lambda: datetime(2026, 8, 7, 13, 0, tzinfo=ZoneInfo("America/New_York")),
    )


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.mark.asyncio
async def test_definitive_indoor_no_open_terminates_retrieval_and_resets_next_turn(monkeypatch):
    _patch_cooling(monkeypatch, [_row("Indoor Option", "Indoor", closed=True)])
    calls: list[str] = []

    async def web_search(_args: dict, _ctx) -> str:
        calls.append("web_search")
        return "web result"

    cooling_tool = cooling.get_tools()[0]
    web_tool = Tool("web_search", "Search", {"type": "object", "properties": {}}, web_search)

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        latest_user = max(
            index
            for index, message in enumerate(messages)
            if any(isinstance(part, UserPromptPart) for part in message.parts)
        )
        returns = _tool_returns(messages[latest_user:])
        if not returns:
            calls.append("cool_options_lookup")
            return ModelResponse([
                ToolCallPart(
                    "cool_options_lookup",
                    {"near": "Times Square", "kind": "indoor"},
                    f"cool-{len(calls)}",
                )
            ])
        if returns[-1].tool_name == "cool_options_lookup":
            result = str(returns[-1].content)
            match = re.search(r"\{cite:([^}]+)\}", result)
            if match is None:
                return ModelResponse([TextPart("model prose")])
            return ModelResponse([ToolCallPart(
                "grounded_answer",
                {"grounded_blocks": [{"text": result, "citation_ids": [match.group(1)]}]},
                "answer-1",
            )])
        return ModelResponse([TextPart("model prose")])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={cooling_tool.name: cooling_tool, web_tool.name: web_tool},
        structured_grounding=True,
        crisis_screen=lambda _turns: _screen_zh(),
    ).conversation()

    first = await conversation.send("现在室内 Cool Options 开放吗？")
    second = await conversation.send("换个地址再查一次")

    assert calls.count("cool_options_lookup") == 2
    assert calls.count("web_search") == 0
    assert "目前没有确认开放的室内 Cool Options" in first.text
    assert first.usage["executed_tool_calls"] == ["cool_options_lookup"]
    assert first.citations
    assert first.usage["safety_model"] == "test/safety"
    assert first.diagnostics["safety_language"] == "zh"
    assert second.usage["executed_tool_calls"].count("cool_options_lookup") == 1


async def _screen_zh():
    return type("Screen", (), {
        "language": "zh",
        "risk": "none",
        "model": "test/safety",
        "input_tokens": 1,
        "output_tokens": 1,
        "cached_input_tokens": 0,
        "requests": 1,
        "cost_usd": 0.0,
        "latency_ms": 1.0,
    })()


@pytest.mark.asyncio
async def test_cooling_center_no_open_allows_indoor_fallback(monkeypatch):
    rows = [_row("Cooling Center", "Outdoor", closed=True), _row("Indoor Option", "Indoor", closed=False)]
    calls: list[dict] = []

    async def cooling_handler(args: dict, ctx):
        calls.append(args.copy())
        return await cooling.get_tools()[0].handler(args, ctx)

    tool = Tool(
        "cool_options_lookup",
        "Find cooling options",
        cooling.get_tools()[0].parameters,
        cooling_handler,
    )

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = _tool_returns(messages)
        if not returns:
            return ModelResponse([ToolCallPart("cool_options_lookup", {"near": "Times Square", "kind": "cooling_center"}, "center-1")])
        if returns[-1].tool_name == "cool_options_lookup" and len(calls) == 1:
            return ModelResponse([ToolCallPart("cool_options_lookup", {"near": "Times Square", "kind": "indoor"}, "indoor-1")])
        return ModelResponse([TextPart("model prose")])

    _patch_cooling(monkeypatch, rows)
    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={tool.name: tool},
        guard_grounding=False,
    ).run("Where can I cool off?")

    assert [call["kind"] for call in calls] == ["cooling_center", "indoor"]
    assert "No current indoor Cool Options" not in result.text


def test_all_cooling_terminal_is_localized_in_chinese():
    message = "No current Cool Options are confirmed open now. I cannot safely recommend a destination."

    assert localize(message, "zh") == "目前没有确认开放的 Cool Options。我无法安全地推荐目的地。"


@pytest.mark.asyncio
async def test_resumed_approval_preserves_chinese_safety_language():
    async def handler(_args: dict, ctx: ToolContext) -> str:
        ctx.cooling_terminal_result = (
            "No current Cool Options are confirmed open now. I cannot safely recommend a destination."
        )
        return "done"

    approval_tool = Tool(
        "approved_lookup",
        "Approved lookup",
        {"type": "object", "properties": {}},
        handler,
        read_only=False,
        requires_approval=True,
    )

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse([TextPart("done")])
        return ModelResponse([ToolCallPart("approved_lookup", {}, "approval-1")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={approval_tool.name: approval_tool},
        guard_grounding=False,
        crisis_screen=lambda _turns: _screen_zh(),
    )
    conversation = runtime.conversation()
    pending = await conversation.send("现在查一下")

    assert pending.status == "approval_required"
    result = await conversation.resume_approvals({"approval-1": True})

    assert result.text == "目前没有确认开放的 Cool Options。我无法安全地推荐目的地。"


@pytest.mark.asyncio
async def test_all_cooling_terminal_cites_every_evaluated_row(monkeypatch):
    rows = [_row("Indoor Option", "Indoor", closed=True), _row("Cooling Center", "Outdoor", closed=True)]
    _patch_cooling(monkeypatch, rows)
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]), user_turns=("查找所有 Cool Options",))

    result = await cooling.get_tools()[0].handler(
        {"near": "Times Square", "kind": "all"},
        ctx,
    )

    assert "No current Cool Options" in result
    terminal = citations.mapping()[ctx.cooling_terminal_citation_ids[0]]
    assert terminal["provenance"].get("snapshot", {}).get("rows") == [
        {
            "record_id": "Indoor Option",
            "snapshot": rows[0],
            "content_hash": content_hash(rows[0]),
            "open_now": False,
        },
        {
            "record_id": "Cooling Center",
            "snapshot": rows[1],
            "content_hash": content_hash(rows[1]),
            "open_now": False,
        },
    ]
    assert len(ctx.cooling_terminal_citation_ids) == 1
    assert set(ctx.cooling_terminal_citation_ids) == set(citations.mapping())
    citation = terminal
    assert citation["snippet"] == result
    assert citation["provenance"]["derivation"]["open_now"] == [
        {"record_id": "Indoor Option", "value": False},
        {"record_id": "Cooling Center", "value": False},
    ]
