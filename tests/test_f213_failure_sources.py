from types import SimpleNamespace

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from heynyc.channels.format import render
from heynyc.core.citations import CitationRegistry
from heynyc.core.pydantic_runtime import PydanticRunFailure, PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.runtime import (
    TEMPORARY_FAILURE_FALLBACK,
    _degraded_failure_text,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext


def _rendered(text: str, citations: CitationRegistry) -> str:
    result = SimpleNamespace(
        text=text,
        citations=citations.mapping(),
        diagnostics={},
        action_links=(),
    )
    return "\n".join(render(result, "sms_twilio"))


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
    rendered = _rendered(text, citations)

    assert "Health help" in text
    assert "https://www.nyc.gov/site/doh/health/health-topics.page" in rendered
    assert old_id not in text
    assert "https://www.nyc.gov/site/immigrants/index.page" not in rendered


def test_failure_source_list_does_not_dump_truncated_web_excerpts() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/help",
        title="Tenant help",
        kind="WEB",
        snippet=(
            "##### **Call 311 and ask for the Tenant Helpline.** "
            "[Tenant Bill of Rights](https://www.nyc.gov/rights) "
            + "more guidance " * 30
        ),
        provenance={"evidence_grade": "authoritative"},
    )

    text = _degraded_failure_text(TEMPORARY_FAILURE_FALLBACK, citations)
    rendered = _rendered(text, citations)

    assert "Tenant help" in text
    assert "https://www.nyc.gov/help" in rendered
    assert "Call 311 and ask for the Tenant Helpline." in text
    assert "Tenant Bill of Rights" not in text
    assert "[Tenant Bill of Rights" not in text
    assert "https://www.nyc.gov/rights" not in text
    assert "#####" not in text
    assert "**" not in text


def test_failure_source_list_keeps_distinct_evidence_from_one_page() -> None:
    citations = CitationRegistry()
    for snippet in ("PACE providers can help.", "The work rule is 80 hours per month."):
        citations.register(
            "https://www.nyc.gov/snap",
            title="SNAP work rules",
            kind="WEB",
            snippet=snippet,
            provenance={"evidence_grade": "authoritative"},
        )

    text = _degraded_failure_text(TEMPORARY_FAILURE_FALLBACK, citations)

    assert "PACE providers can help." in text
    assert "The work rule is 80 hours per month." in text


def test_failure_source_list_does_not_expose_internal_evidence_labels() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/snap",
        title="SNAP work rules",
        kind="WEB",
        snippet="The work rule is 80 hours per month.",
        provenance={"evidence_grade": "authoritative_excerpt"},
    )

    text = _degraded_failure_text(TEMPORARY_FAILURE_FALLBACK, citations)

    assert "SNAP work rules" in text
    assert "Official search excerpt" not in text
    assert "Unverified search result" not in text


def test_failure_source_list_does_not_downgrade_better_evidence_for_same_url() -> None:
    citations = CitationRegistry()
    url = "https://www.nyc.gov/html/dot/html/pedestrians/summerstreets.shtml"
    citations.register(
        url,
        title="Unavailable source",
        kind="WEB",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )
    excerpt_id = citations.register(
        url,
        title="NYC DOT Summer Streets",
        kind="WEB",
        snippet="Brooklyn and The Bronx on August 22 from 9 a.m. to 5 p.m.",
        provenance={
            "evidence_grade": "authoritative_excerpt",
            "source_tier": "authoritative",
        },
    )

    text = _degraded_failure_text(
        f"The final events run from 9 a.m. to 5 p.m. {{cite:{excerpt_id}}}",
        citations,
    )

    assert "Unverified source" not in text
    assert text.count(url) == 0


def test_failure_source_list_keeps_an_unavailable_different_source() -> None:
    citations = CitationRegistry()
    cited_id = citations.register(
        "https://www.nyc.gov/working",
        title="Working source",
        kind="WEB",
        snippet="Current guidance.",
        provenance={"evidence_grade": "authoritative_excerpt"},
    )
    citations.register(
        "https://www.nyc.gov/unavailable",
        title="Unavailable source",
        kind="WEB",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )

    text = _degraded_failure_text(
        f"Current guidance. {{cite:{cited_id}}}",
        citations,
    )

    assert "https://www.nyc.gov/unavailable" in _rendered(text, citations)


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
    conversation.state.citations.register(
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
    rendered = "\n".join(render(caught.value.partial_result, "sms_twilio"))
    assert old_url not in rendered
    assert current_url in rendered
