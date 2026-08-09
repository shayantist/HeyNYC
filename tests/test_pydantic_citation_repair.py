import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import (
    GroundedAnswer,
    GroundedBlock,
    _grounded_block_text,
    _legacy_citation_ids,
    _render_grounded_answer,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def test_matching_citation_marker_variants_are_repaired_without_a_retry():
    block = GroundedBlock(
        text="Benefits may change. { CITE : s1 }",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change."
    assert _render_grounded_answer(GroundedAnswer(grounded_blocks=[block])) == (
        "Benefits may change. {cite:S1}"
    )


def test_matching_unclosed_citation_marker_is_repaired_without_a_retry():
    block = GroundedBlock(
        text="Benefits may change. {cite:S1",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change."
    assert _render_grounded_answer(GroundedAnswer(grounded_blocks=[block])) == (
        "Benefits may change. {cite:S1}"
    )


def test_unknown_citation_marker_is_not_repaired():
    block = GroundedBlock(
        text="Benefits may change. {cite:S2}",
        citation_ids=["S1"],
    )

    assert _grounded_block_text(block) == "Benefits may change. {cite:S2}"
    assert _legacy_citation_ids(_grounded_block_text(block)) == ["S2"]


@pytest.mark.parametrize("marker", ["{ CITE : s1 }", "{cite:S1"])
async def test_matching_marker_variant_does_not_consume_an_output_retry(marker: str):
    async def source(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official guidance",
            kind="WEB",
            snippet="Benefits may change.",
        )
        return f"Benefits may change. {{cite:{citation_id}}}"

    model_calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse([ToolCallPart("guidance", {}, "guidance-1")])
        return ModelResponse([
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "grounded_blocks": [{
                        "text": f"Benefits may change. {marker}",
                        "citation_ids": ["S1"],
                    }]
                },
                "final-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "guidance": Tool(
                name="guidance",
                description="Get current official guidance",
                parameters={"type": "object", "properties": {}},
                handler=source,
            )
        },
        structured_grounding=True,
        guard_grounding=True,
    ).run("Could my benefits change?")

    assert model_calls == 2
    assert result.text == "Benefits may change. {cite:S1}"
    assert result.diagnostics["validation_rejections"] == []
