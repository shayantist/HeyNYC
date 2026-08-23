from types import SimpleNamespace

from heynyc.core.citations import CitationRegistry
from heynyc.core.grounding import check_grounding
from heynyc.core.pydantic_runtime.projection import GroundedAnswer, GroundedBlock
from heynyc.core.pydantic_runtime.tools import ResponsePriorityCapability
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext


async def test_priority_uses_structured_citation_not_text_anchor() -> None:
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
    deps.response_priority_citation_ids.add(priority_id)
    output = GroundedAnswer(
        grounded_blocks=[
            GroundedBlock(
                text="Use NYC FoodHelp now for immediate food help.",
                citation_ids=[priority_id],
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


async def test_f200_priority_block_is_moved_ahead_without_a_model_retry() -> None:
    deps = ToolContext(citations=CitationRegistry(), registry=Registry([]))
    priority_id = deps.citations.register(
        "https://finder.nyc.gov/foodhelp/",
        title="Immediate food help",
        snippet="Call 311 for food tonight.",
        provenance={"derivation": {"response_priority_anchors": ["311"]}},
    )
    pantry_id = deps.citations.register(
        "https://finder.nyc.gov/foodhelp/pantry",
        title="Pantry row",
        snippet="Pantry hours are 10 AM to 6 PM.",
    )
    deps.response_priority_citation_ids.add(priority_id)
    pantry = GroundedBlock(
        text="The pantry lists hours from 10 AM to 6 PM.",
        citation_ids=[pantry_id],
    )
    immediate = GroundedBlock(
        text="Call 311 now for immediate food help.",
        citation_ids=[priority_id],
    )
    output = GroundedAnswer(grounded_blocks=[pantry, immediate])

    repaired = await ResponsePriorityCapability().after_output_validate(
        SimpleNamespace(deps=deps),
        output_context=None,
        output=output,
    )

    assert repaired.grounded_blocks == [immediate, pantry]
    assert deps.validation_rejections == []


def test_f194_priority_metadata_is_not_claim_support_evidence() -> None:
    citations = {
        "S1": {
            "title": "Immediate food help",
            "kind": "DATA",
            "snippet": "No pantry availability is confirmed tonight.",
            "provenance": {
                "snapshot": {"scheduled_open_nearby": 0},
                "derivation": {"response_priority_anchors": ["311"]},
            },
        }
    }

    result = check_grounding("Call 311 for immediate food help. {cite:S1}", citations)

    assert result is not None
    assert result.blocking


def test_f200_grounded_block_schema_requires_source_homogeneous_blocks() -> None:
    schema = GroundedBlock.model_json_schema()["properties"]

    assert "separate block" in schema["text"]["description"].lower()
    assert "all sources" in schema["citation_ids"]["description"].lower()
