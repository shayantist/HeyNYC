from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule
from heynyc.core.pydantic_runtime.projection import GroundedBlock
from heynyc.core.pydantic_runtime.safety import build_scope_screen
from heynyc.core.registry import Registry
from heynyc.core.tools.base import ToolContext
from heynyc.modules.events.tools import EventQuery
from heynyc.modules.nyc311_status import tools as nyc311


async def test_scope_screen_receives_the_complete_module_description() -> None:
    marker = "important scope detail after the old cutoff"
    description = "intro " * 30 + marker

    async def classify(messages, info: AgentInfo) -> ModelResponse:
        assert marker in str(messages)
        output = info.output_tools[0]
        return ModelResponse([
            ToolCallPart(
                output.name,
                {"event_turn": None, "modules": ["housing"], "situations": []},
                "scope-1",
            )
        ])

    screen = build_scope_screen(
        FunctionModel(classify),
        model_name="test/scope",
        registry=Registry([ServiceModule(
            name="housing",
            description=description,
        )]),
    )

    result = await screen(("I need help",))

    assert result.modules == ("housing",)


def test_grounded_block_accepts_every_supporting_citation() -> None:
    citation_ids = [f"S{index}" for index in range(1, 13)]

    block = GroundedBlock(text="One compound sourced claim.", citation_ids=citation_ids)

    assert block.citation_ids == citation_ids


async def test_311_search_preserves_every_resident_supplied_term(monkeypatch) -> None:
    async def lookup(terms, *_args):
        return ",".join(terms)

    monkeypatch.setattr(nyc311, "_lookup_area", lookup)
    ctx = ToolContext(citations=CitationRegistry(), registry=Registry([]))

    result = await nyc311._search_311_complaints(
        {"complaint_terms": ["noise", "music", "party", "construction"]},
        ctx,
    )

    assert result == "noise,music,party,construction"


def test_event_query_accepts_a_resident_requested_count_above_ten() -> None:
    query = EventQuery.model_validate({"max_results": 25})

    assert query.max_results == 25
