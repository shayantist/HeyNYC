from __future__ import annotations

from heynyc.core.tools.base import Tool


async def _noop(args, ctx):
    return ""


def _tool(**kw):
    base = dict(
        name="demo",
        description="d",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=_noop,
    )
    base.update(kw)
    return Tool(**base)


def test_schema_injects_additional_properties_false():
    # Strict-schema best practice: model can't improvise extra args.
    fn = _tool().schema()["function"]
    assert fn["parameters"]["additionalProperties"] is False
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
