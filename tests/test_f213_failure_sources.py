import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime import PydanticRunFailure, PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import (
    TEMPORARY_FAILURE_FALLBACK,
    _degraded_failure_text,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def test_f213_failure_links_only_include_sources_reached_this_turn() -> None:
    citations = CitationRegistry()
    old_id = citations.register(
        "https://www.nyc.gov/site/immigrants/index.page",
        title="Immigration help",
        kind="WEB",
        snippet="Help from an earlier turn",
        provenance={"evidence_grade": "authoritative"},
    )
    current_id = citations.register(
        "https://www.nyc.gov/site/doh/health/health-topics.page",
        title="Health help",
        kind="WEB",
        snippet="Help reached on this turn",
        provenance={"evidence_grade": "authoritative"},
    )

    text = _degraded_failure_text(
        TEMPORARY_FAILURE_FALLBACK,
        citations,
        citation_ids={current_id},
    )

    assert "Health help" in text
    assert old_id not in text
    assert "Immigration help" not in text


@pytest.mark.asyncio
async def test_f213_approval_resume_failure_excludes_preexisting_sources(
    monkeypatch,
) -> None:
    async def act(_args: dict, _ctx: ToolContext) -> str:
        return "done"

    runtime = PydanticRuntimeAdapter(
        TestModel(call_tools=["act"], custom_output_text="Finished"),
        registry=Registry([]),
        tools={
            "act": Tool(
                name="act",
                description="Complete an approved action",
                parameters={"type": "object", "properties": {}},
                handler=act,
                requires_approval=True,
            )
        },
        guard_grounding=False,
    )
    conversation = runtime.conversation()
    pending = await conversation.send("Do it")
    old_url = "https://www.nyc.gov/site/immigrants/index.page"
    current_url = "https://www.nyc.gov/site/doh/health/health-topics.page"
    conversation._citations.register(
        old_url,
        title="Immigration help",
        kind="WEB",
        snippet="Help from an earlier turn",
        provenance={"evidence_grade": "authoritative"},
    )

    async def fail(*_args, **kwargs):
        kwargs["deps"].citations.register(
            current_url,
            title="Health help",
            kind="WEB",
            snippet="Help reached during approval resume",
            provenance={"evidence_grade": "authoritative"},
        )
        raise UnexpectedModelBehavior("broken output")

    monkeypatch.setattr(runtime._agent, "run", fail)

    with pytest.raises(PydanticRunFailure) as caught:
        await conversation.resume_approvals(
            {call_id: True for call_id in conversation.pending_approvals}
        )

    assert pending.status == "approval_required"
    assert old_url not in caught.value.partial_result.text
    assert current_url in caught.value.partial_result.text
