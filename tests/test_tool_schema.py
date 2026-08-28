from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from heynyc.core.citations import CitationRegistry
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext, ToolInput


class DemoInput(BaseModel):
    q: str = Field(description="Search text")


async def _noop(args: DemoInput, ctx):
    return ""


def _tool(**kw):
    base = dict(
        name="demo",
        description="d",
        input_type=DemoInput,
        handler=_noop,
    )
    base.update(kw)
    return Tool(**base)


def test_schema_injects_additional_properties_false():
    fn = _tool().schema()["function"]
    assert fn["parameters"]["additionalProperties"] is False
    assert fn["parameters"]["properties"]["q"] == {
        "description": "Search text",
        "title": "Q",
        "type": "string",
    }
    assert "strict" not in fn  # off by default


def test_schema_strict_flag_emitted():
    fn = _tool(strict=True).schema()["function"]
    assert fn["strict"] is True


def test_to_mcp_emits_spec_shape_with_annotations():
    mcp = _tool(open_world=True, read_only=True).to_mcp()
    assert mcp["name"] == "demo"
    assert mcp["inputSchema"]["additionalProperties"] is False  # same strict schema
    ann = mcp["annotations"]
    assert ann["readOnlyHint"] is True
    assert ann["openWorldHint"] is True
    assert ann["destructiveHint"] is False
    assert ann["idempotentHint"] is True
    assert ann["title"] == "demo"


def test_core_tools_carry_honest_annotations():
    # The real toolbox tags external tools open-world, curated index closed-world.
    from heynyc.core.registry import Registry
    from heynyc.core.tools import build_toolbox

    tools = build_toolbox(Registry([]))
    assert tools["geocode"].open_world is True
    assert tools["web_search"].open_world is True
    # every tool serializes to both standard shapes without error
    for t in tools.values():
        assert t.schema()["function"]["parameters"]["additionalProperties"] is False
        assert set(t.to_mcp()["annotations"]) >= {"readOnlyHint", "openWorldHint", "idempotentHint"}


def test_service_tools_use_pydantic_as_their_input_contract():
    from heynyc.core.registry import Registry
    from heynyc.core.tools import build_toolbox

    tools = build_toolbox(Registry.discover(Path("heynyc/modules"))).values()

    assert all(tool.input_type is not None for tool in tools)
    assert all(tool.input_type.model_config.get("extra") == "forbid" for tool in tools)


async def test_tool_invocation_validates_once_and_passes_the_typed_request():
    class CountInput(ToolInput):
        count: int

    seen = []

    async def handler(request: CountInput, _ctx: ToolContext):
        seen.append(request)
        return request["count"]

    tool = Tool(name="count", description="Count", input_type=CountInput, handler=handler)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    assert await tool.invoke({"count": "2"}, ctx) == 2
    assert isinstance(seen[0], CountInput)
    with pytest.raises(ValidationError):
        await tool.invoke({"count": 2, "unexpected": True}, ctx)


async def test_tool_invocation_rejects_unknown_fields_for_plain_base_models():
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    with pytest.raises(ValidationError):
        await _tool().invoke({"q": "hello", "unexpected": True}, ctx)


async def test_tool_invocation_returns_expected_operational_failure() -> None:
    from heynyc.core.tools.base import ToolFailureError

    async def handler(_request: ToolInput, _ctx: ToolContext):
        raise ToolFailureError(
            status="unavailable",
            reason="provider offline",
            retryable=True,
        )

    tool = Tool(name="fails", description="Fails", handler=handler)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await tool.invoke({}, ctx)

    assert result.status == "unavailable"
    assert result.reason == "provider offline"
