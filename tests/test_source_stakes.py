from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from heynyc.channels.format import render
from heynyc.core.agent import AgentResult
from heynyc.core.citations import CitationRegistry, used_unverified_citations
from heynyc.core.manifest import ServiceModule, SituationHint
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import GroundedAnswer, GroundedBlock
from heynyc.core.pydantic_runtime.tools import build_module_capabilities
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.web_search import web_search_tools
from heynyc.eval.cases import load_cases


def test_preferred_domain_audit_records_only_primary_publishers() -> None:
    registry = Registry.discover(Path("heynyc/modules"))
    tiers = registry.source_tiers()

    assert tiers["511ny.org"][0] == "authoritative"
    assert tiers["nyccare.nyc"][0] == "authoritative"
    assert tiers["nychealthandhospitals.org"][0] == "authoritative"
    assert tiers["poison.org"][0] == "authoritative"
    assert "everbridge.net" not in tiers
    assert "services6.arcgis.com" not in tiers


def test_source_tiers_also_supply_preferred_domains() -> None:
    registry = Registry([
        ServiceModule(
            name="events",
            source_tiers={"editorial": ["example.org"]},
        ),
    ])

    assert registry.allowlist() == ["example.org"]


async def test_low_stakes_unknown_search_excerpt_is_answer_grade() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Neighborhood event roundup",
            "url": "https://new-local-site.example/weekend",
            "snippet": "The free outdoor movie starts at 7 p.m.",
        }]

    tool = web_search_tools([], search_fn=search)[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        allow_unverified_search_excerpts=True,
    )

    await tool.handler({"query": "free movies this weekend"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "search_excerpt",
        "source_tier": "unverified",
    }


async def test_low_stakes_community_search_excerpt_is_answer_grade() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Neighborhood event listing",
            "url": "https://eventbrite.com/e/neighborhood-event",
            "snippet": "The neighborhood event starts Friday at 7 p.m.",
        }]

    tool = web_search_tools(
        ["eventbrite.com"],
        source_tiers={"eventbrite.com": ("community", "events")},
        search_fn=search,
    )[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        allow_unverified_search_excerpts=True,
    )

    await tool.handler({"query": "neighborhood events Friday"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "search_excerpt",
        "source_tier": "community",
    }


async def test_high_stakes_community_search_excerpt_stays_discovery_only() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Community benefits post",
            "url": "https://eventbrite.com/e/benefits-advice",
            "snippet": "A community post describes benefit eligibility.",
        }]

    tool = web_search_tools(
        ["eventbrite.com"],
        source_tiers={"eventbrite.com": ("community", "events")},
        search_fn=search,
    )[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        allow_unverified_search_excerpts=True,
        current_turn_high_stakes=True,
    )

    await tool.handler({"query": "benefit eligibility"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
    }


async def test_high_stakes_unknown_search_excerpt_stays_discovery_only() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Unknown benefits post",
            "url": "https://unknown.example/benefits-advice",
            "snippet": "An unknown post describes benefit eligibility.",
        }]

    tool = web_search_tools([], search_fn=search)[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        allow_unverified_search_excerpts=True,
        current_turn_high_stakes=True,
    )

    await tool.handler({"query": "benefit eligibility"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"] == {
        "evidence_grade": "discovery",
        "source_tier": "unverified",
    }


def test_high_stakes_guard_catches_fetched_unverified_evidence() -> None:
    citations = {
        "S1": {"provenance": {"evidence_grade": "fetched", "source_tier": "unverified"}},
        "S2": {"provenance": {"evidence_grade": "fetched", "source_tier": "authoritative"}},
    }

    assert used_unverified_citations("Claim {cite:S1} Official {cite:S2}", citations) == ["S1"]


async def test_low_stakes_archived_excerpt_stays_discovery_only() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Archived neighborhood event",
            "url": "https://new-local-site.example/archive/weekend",
            "snippet": "The free outdoor movie starts at 7 p.m.",
        }]

    tool = web_search_tools([], search_fn=search)[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        allow_unverified_search_excerpts=True,
    )

    await tool.handler({"query": "free movies this weekend"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"]["evidence_grade"] == "discovery"


async def test_event_label_alone_cannot_relax_source_policy() -> None:
    async def search(_query, _domains, **_kwargs):
        return [{
            "title": "Neighborhood event roundup",
            "url": "https://new-local-site.example/weekend",
            "snippet": "The free outdoor movie starts at 7 p.m.",
        }]

    tool = web_search_tools([], search_fn=search)[0]
    ctx = ToolContext(
        citations=CitationRegistry(),
        registry=Registry([]),
        event_turn="discovery",
    )

    await tool.handler({"query": "free movies this weekend"}, ctx)

    assert ctx.citations.mapping()["S1"]["provenance"]["evidence_grade"] == "discovery"


def test_pydantic_source_policy_uses_the_current_capability() -> None:
    registry = Registry([
        ServiceModule(name="events"),
        ServiceModule(name="benefits"),
    ])

    async def handler(_args, _ctx):
        return "ok"

    web_search = Tool(
        name="web_search",
        description="search",
        handler=handler,
    )
    adapted, _capabilities = build_module_capabilities(
        registry,
        {"web_search": web_search},
    )
    tool = adapted[0]
    deps = ToolContext(
        citations=CitationRegistry(),
        registry=registry,
        allow_unverified_search_excerpts=True,
    )
    context = SimpleNamespace(
        deps=deps,
        loaded_capability_ids={"events"},
        messages=[
            ModelRequest(parts=[UserPromptPart("What events are on?")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "events"}, tool_call_id="load-events"
            )]),
        ],
    )
    definition = ToolDefinition(
        name="web_search",
        description="search",
        parameters_json_schema={"type": "object", "properties": {}},
    )

    assert tool.prepare(context, definition) is definition
    assert deps.allow_unverified_search_excerpts is True

    deps.current_turn_high_stakes = True

    assert tool.prepare(context, definition) is definition
    assert deps.allow_unverified_search_excerpts is False


async def test_high_stakes_output_requires_cited_claim_blocks() -> None:
    registry = Registry([
        ServiceModule(
            name="benefits",
            situations=[SituationHint(
                name="appeal",
                definition="Appeal a benefits decision.",
                high_stakes=True,
            )],
        )
    ])
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=registry,
        tools={},
        structured_grounding=True,
        use_module_capabilities=True,
    )
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/site/hra/help/fair-hearings.page",
        title="HRA fair hearings",
        kind="WEB",
        snippet="Request a fair hearing from HRA.",
        provenance={
            "evidence_grade": "authoritative",
            "source_tier": "authoritative",
        },
    )
    ctx = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=registry, query="How do I appeal?"),
        messages=[
            ModelRequest(parts=[UserPromptPart("How do I appeal?")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "benefits-appeal"},
                tool_call_id="load-appeal",
            )]),
        ],
    )

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(
            ctx,
            "You should appeal online.",
        )

    assert ctx.deps.validation_rejections[-1]["stage"] == "high_stakes_format"
    assert await runtime._validate_grounding(
        ctx,
        GroundedAnswer(grounded_blocks=[GroundedBlock(
            text="Request a fair hearing from HRA.",
            citation_ids=["S1"],
        )]),
    ) == GroundedAnswer(grounded_blocks=[GroundedBlock(
        text="Request a fair hearing from HRA.",
        citation_ids=["S1"],
    )])


async def test_loaded_root_capability_inherits_its_high_stakes_situations() -> None:
    registry = Registry([
        ServiceModule(
            name="benefits",
            situations=[SituationHint(
                name="appeal",
                definition="Benefits appeal guidance",
                high_stakes=True,
            )],
        )
    ])
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=registry,
        tools={},
        structured_grounding=True,
        use_module_capabilities=True,
    )
    ctx = SimpleNamespace(
        deps=ToolContext(citations=CitationRegistry(), registry=registry),
        messages=[
            ModelRequest(parts=[UserPromptPart("Can I appeal?")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "benefits"},
                tool_call_id="load-benefits",
            )]),
        ],
    )

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(ctx, "You are definitely eligible to appeal.")

    assert ctx.deps.validation_rejections[-1]["stage"] == "high_stakes_format"


async def test_high_stakes_claim_rejects_discovery_only_evidence() -> None:
    registry = Registry([
        ServiceModule(
            name="benefits",
            situations=[SituationHint(
                name="eligibility",
                definition="Benefits eligibility guidance",
                high_stakes=True,
            )],
        )
    ])
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=registry,
        tools={},
        structured_grounding=True,
        use_module_capabilities=True,
    )
    citations = CitationRegistry()
    citations.register(
        "https://example.org/benefits",
        snippet="A search result says every applicant qualifies.",
        provenance={"evidence_grade": "discovery", "source_tier": "unverified"},
    )
    ctx = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=registry),
        messages=[
            ModelRequest(parts=[UserPromptPart("Do I qualify?")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "benefits"},
                tool_call_id="load-benefits",
            )]),
        ],
    )

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(
            ctx,
            GroundedAnswer(grounded_blocks=[GroundedBlock(
                text="Every applicant qualifies.",
                citation_ids=["S1"],
            )]),
        )

    assert ctx.deps.validation_rejections[-1]["stage"] == "discovery_only"


async def test_governed_tool_use_marks_the_turn_high_stakes() -> None:
    registry = Registry([
        ServiceModule(
            name="benefits",
            situations=[SituationHint(
                name="eligibility",
                definition="Benefits eligibility guidance",
                high_stakes=True,
            )],
        )
    ])
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=registry,
        tools={"check_benefits": Tool(
            name="check_benefits",
            description="Check benefits",
            handler=lambda _args, _ctx: None,
            module="benefits",
        )},
        structured_grounding=True,
        use_module_capabilities=True,
    )
    ctx = SimpleNamespace(
        deps=ToolContext(
            citations=CitationRegistry(),
            registry=registry,
            tool_runs=[{"tool": "check_benefits", "status": "success"}],
        ),
        messages=[ModelRequest(parts=[UserPromptPart("Do I qualify?")])],
    )

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(ctx, "You definitely qualify.")

    assert ctx.deps.validation_rejections[-1]["stage"] == "high_stakes_format"


async def test_high_stakes_mixed_turn_preserves_a_sourced_low_stakes_lead() -> None:
    registry = Registry([
        ServiceModule(
            name="benefits",
            situations=[SituationHint(
                name="eligibility",
                definition="Benefits eligibility guidance",
                high_stakes=True,
            )],
        )
    ])
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=registry,
        tools={},
        structured_grounding=True,
        use_module_capabilities=True,
    )
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/site/hra/help/snap-benefits-food-program.page",
        snippet="Apply for SNAP through ACCESS HRA.",
        provenance={"evidence_grade": "authoritative", "source_tier": "authoritative"},
    )
    citations.register(
        "https://example.org/finance-event",
        snippet="Finance networking event Thursday from 6 to 9 PM.",
        provenance={"evidence_grade": "discovery", "source_tier": "community"},
    )
    ctx = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=registry),
        messages=[
            ModelRequest(parts=[UserPromptPart("Help with SNAP and an event tonight")]),
            ModelResponse(parts=[LoadCapabilityCallPart(
                args={"id": "benefits"},
                tool_call_id="load-benefits",
            )]),
        ],
    )
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(text="Apply for SNAP through ACCESS HRA.", citation_ids=["S1"]),
        GroundedBlock(
            kind="lead",
            text="A community source lists a finance networking event Thursday from 6 to 9 PM.",
            citation_ids=["S2"],
        ),
    ])

    assert await runtime._validate_grounding(ctx, output) == output


async def test_grounded_blocks_keep_exact_dates_bound_to_their_declared_sources() -> None:
    runtime = PydanticRuntimeAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse([])),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    )
    citations = CitationRegistry()
    citations.register(
        "https://events.example/august",
        snippet="Finance August Networking Event Thu, Aug 27th, 2026, 6:00 PM to 9:00 PM.",
        provenance={"evidence_grade": "search_excerpt"},
    )
    citations.register(
        "https://events.example/september",
        snippet="Join us Wednesday, September 2nd, 2026, at 6:00 PM ET.",
        provenance={"evidence_grade": "search_excerpt"},
    )
    ctx = SimpleNamespace(
        deps=ToolContext(citations=citations, registry=Registry([])),
        messages=[ModelRequest(parts=[UserPromptPart("What events did you find?")])],
    )
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(
            text="The August event is on August 27, 2026, from 6:00 PM to 9:00 PM. https://events.example/august",
            citation_ids=["S1"],
        ),
        GroundedBlock(
            text="The next listing is September 2, 2026, at 6:00 PM ET. https://events.example/september",
            citation_ids=["S2"],
        ),
    ])

    assert await runtime._validate_grounding(ctx, output) == output


async def test_pydantic_scope_checklist_sets_current_source_policy() -> None:
    registry = Registry([ServiceModule(name="events")])

    async def search(_args: dict, ctx: ToolContext) -> str:
        assert ctx.current_turn_modules == frozenset({"events"})
        assert ctx.allow_unverified_search_excerpts is True
        citation_id = ctx.citations.register(
            "https://unknown.example/event",
            title="Event recap",
            kind="WEB",
            snippet="The event is Saturday at 7 p.m. and admission is free.",
            provenance={
                "evidence_grade": "search_excerpt",
                "source_tier": "unverified",
            },
        )
        return f"Event details. {{cite:{citation_id}}}"

    async def scope(_turns: tuple[str, ...]):
        return SimpleNamespace(
            event_turn="discovery",
            modules=("events",),
            situations=(),
            model="test/scope",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            requests=1,
            cost_usd=0.0,
            latency_ms=1.0,
            usable=True,
        )

    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("web_search", {}, "search-1")])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {"answer": "The event is Saturday at 7 p.m. and admission is free. {cite:S1}"},
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=registry,
        tools={
            "web_search": Tool(
                name="web_search",
                description="Search the web",
                handler=search,
            )
        },
        structured_grounding=True,
        use_module_capabilities=True,
        scope_screen=scope,
    ).run("Tell me more about that event.")

    assert result.text.endswith("{cite:S1}")
    assert result.usage["scope_model"] == "test/scope"
    message = "\n".join(render(result, "sms"))
    assert "https://unknown.example/event" in message
    assert "verification note" not in message.lower()


async def test_pydantic_untyped_scope_result_keeps_sourced_excerpts_available() -> None:
    async def untyped_scope(_turns: tuple[str, ...]):
        return SimpleNamespace(event_turn=None, modules=(), situations=())

    async def search(_args: dict, ctx: ToolContext) -> str:
        assert ctx.allow_unverified_search_excerpts is True
        return "No excerpt used."

    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("web_search", {}, "search-1")])
        return ModelResponse([
            ToolCallPart("final_answer", {"answer": "I could not verify that."}, "answer-1")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([ServiceModule(name="events")]),
        tools={
            "web_search": Tool(
                name="web_search",
                description="Search the web",
                handler=search,
            )
        },
        structured_grounding=True,
        use_module_capabilities=True,
        scope_screen=untyped_scope,
    ).run("What is happening tonight?")

    assert result.text == "I could not verify that."


def test_source_backed_excerpt_does_not_get_a_blanket_warning() -> None:
    result = AgentResult(
        text="The free outdoor movie starts at 7 p.m. {cite:S1}",
        status="success",
        citations={
            "S1": {
                "url": "https://new-local-site.example/weekend",
                "title": "Neighborhood event roundup",
                "kind": "WEB",
                "snippet": "The free outdoor movie starts at 7 p.m.",
                "provenance": {
                    "evidence_grade": "search_excerpt",
                    "source_tier": "unverified",
                },
            }
        },
    )

    message = "\n".join(render(result, "sms"))

    assert "https://new-local-site.example/weekend" in message
    assert "verification note" not in message.lower()


def test_authoritative_excerpt_does_not_get_unverified_warning() -> None:
    result = AgentResult(
        text="The free outdoor movie starts at 7 p.m. {cite:S1}",
        status="success",
        citations={
            "S1": {
                "url": "https://parks.nyc.gov/events/movie",
                "title": "NYC Parks event",
                "kind": "WEB",
                "snippet": "The free outdoor movie starts at 7 p.m.",
                "provenance": {
                    "evidence_grade": "authoritative_excerpt",
                    "source_tier": "authoritative",
                },
            }
        },
    )

    message = "\n".join(render(result, "sms"))

    assert "search-result excerpt" not in message.lower()


def test_live_source_stakes_cases_pin_the_three_unfinished_paths() -> None:
    cases = {
        case.id: case
        for case in load_cases(Registry.discover(Path("heynyc/modules")))
    }

    low = cases["source_stakes_low_stakes_excerpt"]
    stale = cases["source_stakes_stale_event_to_snap"]
    inverse = cases["source_stakes_high_stakes_inverse"]

    assert low.expect_tools == ["web_search"]
    assert len(low.turns) == 2
    assert "F266" in low.tags
    assert "unverified excerpt" in low.utility_criterion.lower()
    assert len(stale.turns) == 2
    assert stale.turns[-1] == inverse.query
    assert stale.expect_tools == ["web_search"]
    assert inverse.expect_tools == ["web_search"]
    assert "F265" in inverse.tags
    assert "F267" in inverse.tags
    assert stale.harm_category == inverse.harm_category == "specialized_advice"
