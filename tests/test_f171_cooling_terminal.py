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
from heynyc.core.manifest import ServiceModule
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
            calls.append("find_cool_options")
            return ModelResponse([
                ToolCallPart(
                    "find_cool_options",
                    {"near": "Times Square", "kind": "indoor"},
                    f"cool-{len(calls)}",
                )
            ])
        if returns[-1].tool_name == "find_cool_options":
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

    assert calls.count("find_cool_options") == 2
    assert calls.count("web_search") == 0
    assert "目前没有确认开放的室内 Cool Options" in first.text
    assert first.usage["executed_tool_calls"] == ["find_cool_options"]
    assert first.citations
    assert first.usage["safety_model"] == "test/safety"
    assert first.diagnostics["safety_language"] == "zh"
    assert second.usage["executed_tool_calls"].count("find_cool_options") == 1


@pytest.mark.asyncio
async def test_f201_cooling_absence_does_not_discard_another_tools_result():
    async def cooling_handler(_args: dict, ctx: ToolContext) -> str:
        result = "No current Cool Options are confirmed open now."
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/cooling",
            title="Cooling lookup",
            snippet=result,
        )
        ctx.cooling_terminal_result = result
        ctx.cooling_terminal_citation_ids = (citation_id,)
        return f"{result} {{cite:{citation_id}}}"

    async def fountain_handler(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/fountains",
            title="Drinking fountains",
            snippet="Hunts Point Playground has an active drinking fountain.",
        )
        return f"Hunts Point Playground has an active drinking fountain. {{cite:{citation_id}}}"

    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([
                ToolCallPart("find_cool_options", {}, "cooling-1"),
                ToolCallPart("nearest_fountain", {}, "fountain-1"),
            ])
        return ModelResponse([
            TextPart(
                "No current Cool Options are confirmed open now. {cite:S1}\n\n"
                "Hunts Point Playground has an active drinking fountain. {cite:S2}"
            )
        ])

    tools = {
        "find_cool_options": Tool(
            "find_cool_options",
            "Check cooling",
            {"type": "object", "properties": {}},
            cooling_handler,
        ),
        "nearest_fountain": Tool(
            "nearest_fountain",
            "Find water",
            {"type": "object", "properties": {}},
            fountain_handler,
        ),
    }
    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools=tools,
        structured_grounding=True,
    ).run("Where can I cool down and refill water?")

    assert calls == 2
    assert "No current Cool Options" in result.text
    assert "Hunts Point Playground" in result.text


@pytest.mark.asyncio
async def test_cooling_terminal_ignores_capability_discovery_returns():
    calls = 0

    async def cooling_handler(_args: dict, ctx: ToolContext) -> str:
        result = "No current indoor Cool Options are confirmed open now."
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/cooling",
            title="Cooling lookup",
            snippet=result,
        )
        ctx.cooling_terminal_result = result
        ctx.cooling_terminal_citation_ids = (citation_id,)
        return f"{result} {{cite:{citation_id}}}"

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([
                ToolCallPart("load_capability", {"id": "cooling_centers"}, "load-1")
            ])
        if calls == 2:
            return ModelResponse([
                ToolCallPart("search_tools", {"queries": ["cooling_centers"]}, "search-1")
            ])
        if calls == 3:
            return ModelResponse([
                ToolCallPart("find_cool_options", {}, "cooling-1")
            ])
        return ModelResponse([TextPart("model should not run again")])

    tool = Tool(
        "find_cool_options",
        "Check cooling",
        {"type": "object", "properties": {}},
        cooling_handler,
        module="cooling_centers",
    )
    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([
            ServiceModule(
                name="cooling_centers",
                description="Find cooling",
                focus_tools=["find_cool_options"],
            )
        ]),
        tools={tool.name: tool},
        use_module_capabilities=True,
        structured_grounding=True,
    ).run("Where can I cool down indoors?")

    assert calls == 3
    assert result.text.startswith("No current indoor Cool Options")


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
        "find_cool_options",
        "Find cooling options",
        cooling.get_tools()[0].parameters,
        cooling_handler,
    )

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = _tool_returns(messages)
        if not returns:
            return ModelResponse([ToolCallPart("find_cool_options", {"near": "Times Square", "kind": "cooling_center"}, "center-1")])
        if returns[-1].tool_name == "find_cool_options" and len(calls) == 1:
            return ModelResponse([ToolCallPart("find_cool_options", {"near": "Times Square", "kind": "indoor"}, "indoor-1")])
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
    message = (
        "No current Cool Options are confirmed open now. I cannot safely recommend a destination. "
        "Call 311 for help finding a currently open place to cool down. "
        "If anyone has trouble breathing, call 911."
    )

    assert localize(message, "zh") == (
        "目前没有确认开放的 Cool Options。我无法安全地推荐目的地。"
        "请拨打 311，寻求帮助查找目前开放的避暑场所。如果有人呼吸困难，请拨打 911。"
    )
    assert localize(message, "es") == (
        "No se confirmó que haya Cool Options abiertos ahora. No puedo recomendar un lugar con "
        "seguridad. Llama al 311 para que te ayuden a encontrar un lugar abierto donde refrescarte. "
        "Si alguien tiene dificultad para respirar, llama al 911."
    )
    assert localize(message, "bn") == (
        "এখন কোনো Cool Options খোলা আছে বলে নিশ্চিত হওয়া যায়নি। আমি নিরাপদভাবে কোনো গন্তব্য "
        "সুপারিশ করতে পারছি না। এখন খোলা কোনো শীতল জায়গা খুঁজে পেতে সাহায্যের জন্য 311-এ ফোন "
        "করুন। কারও শ্বাসকষ্ট হলে 911-এ ফোন করুন।"
    )


@pytest.mark.asyncio
async def test_f222_no_open_indoor_option_keeps_heat_safety_next_steps(monkeypatch):
    _patch_cooling(monkeypatch, [_row("Indoor Option", "Indoor", closed=True)])
    citations = CitationRegistry()
    ctx = ToolContext(citations=citations, registry=Registry([]))

    result = await cooling.get_tools()[0].handler(
        {"near": "Times Square", "kind": "indoor"},
        ctx,
    )

    assert "311" in result
    assert "trouble breathing" in result
    assert "911" in result
    assert len(ctx.cooling_terminal_citation_ids) == 2
    heat_help = citations.mapping()[ctx.cooling_terminal_citation_ids[1]]
    assert heat_help["provenance"]["snapshot"]["verified_fact"] == cooling._HEAT_HELP


@pytest.mark.asyncio
async def test_f222_open_indoor_option_does_not_append_emergency_backstop(monkeypatch):
    _patch_cooling(monkeypatch, [_row("Indoor Option", "Indoor", closed=False)])
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await cooling.get_tools()[0].handler(
        {"near": "Times Square", "kind": "indoor"},
        ctx,
    )

    assert "Indoor Option site" in result
    assert "Call 311" not in result
    assert "trouble breathing" not in result
    assert "call 911" not in result.casefold()


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
    assert len(ctx.cooling_terminal_citation_ids) == 2
    assert set(ctx.cooling_terminal_citation_ids) == set(citations.mapping())
    citation = terminal
    assert citation["snippet"] == (
        "No current Cool Options are confirmed open now. I cannot safely recommend a destination."
    )
    assert citation["provenance"]["derivation"]["open_now"] == [
        {"record_id": "Indoor Option", "value": False},
        {"record_id": "Cooling Center", "value": False},
    ]
