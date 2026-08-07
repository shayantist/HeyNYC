from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime.projection import GroundedAnswer, GroundedBlock
from heynyc.core.pydantic_runtime.tools import ResponsePriorityCapability
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext


async def test_priority_block_rejects_mixed_lower_priority_evidence() -> None:
    deps = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    priority_id = deps.citations.register(
        "https://finder.nyc.gov/foodhelp/",
        title="Immediate food help",
        snippet="Call 311 or use NYC FoodHelp for food tonight.",
        provenance={
            "derivation": {
                "response_priority_anchors": [
                    "311",
                    "https://finder.nyc.gov/foodhelp/",
                ]
            }
        },
    )
    lower_priority_id = deps.citations.register(
        "https://access.nyc.gov/",
        title="Benefits estimate",
        snippet="The benefits result is an estimate.",
    )
    deps.response_priority_citation_ids.add(priority_id)
    output = GroundedAnswer(
        grounded_blocks=[
            GroundedBlock(
                text="A pantry is listed nearby.",
                citation_ids=[priority_id, lower_priority_id],
            )
        ]
    )

    with pytest.raises(ModelRetry):
        await ResponsePriorityCapability().after_output_validate(
            SimpleNamespace(deps=deps),
            output_context=None,
            output=output,
        )


async def test_priority_block_accepts_declared_action_anchor() -> None:
    deps = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    priority_id = deps.citations.register(
        "https://finder.nyc.gov/foodhelp/",
        title="Immediate food help",
        snippet="Call 311 or use NYC FoodHelp for food tonight.",
        provenance={"derivation": {"response_priority_anchors": ["311"]}},
    )
    lower_priority_id = deps.citations.register(
        "https://access.nyc.gov/",
        title="Benefits estimate",
        snippet="The benefits result is an estimate.",
    )
    deps.response_priority_citation_ids.add(priority_id)
    output = GroundedAnswer(
        grounded_blocks=[
            GroundedBlock(
                text="Call 311 now for food tonight, then we can check SNAP.",
                citation_ids=[priority_id, lower_priority_id],
            )
        ]
    )

    assert (
        await ResponsePriorityCapability().after_output_validate(
            SimpleNamespace(deps=deps),
            output_context=None,
            output=output,
        )
        is output
    )
