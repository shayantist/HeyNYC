from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.core import agent
from heynyc.core.agent import GROUNDING_ABSTAIN_FALLBACK, Agent, ScopeResult
from heynyc.core.citations import CitationRegistry
from heynyc.core.manifest import ServiceModule, SituationHint
from heynyc.core.pydantic_runtime.projection import GroundedAnswer, GroundedBlock
from heynyc.core.pydantic_runtime.runtime import PydanticRuntimeAdapter
from heynyc.core.registry import Registry
from heynyc.core.tools import Tool

QUERY = "আমার বাসায় পাঁচ দিন ধরে গরম পানি নেই। NYC code-এর কোন section এটা cover করে, আর এখন কী করব?"


def _citations() -> CitationRegistry:
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/site/hpd/services-and-information/heat-and-hot-water-information.page",
        title="Heat and Hot Water Information, NYC HPD",
        snippet="File a heat or hot-water complaint by calling 311",
        kind="DOC",
    )
    citations.register(
        "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-60410",
        title="NYC Housing Maintenance Code section 27-2029 (Heat)",
        snippet="The minimum indoor temperature rule is section 27-2029",
        kind="DOC",
    )
    citations.register(
        "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-236495",
        title="NYC Housing Maintenance Code section 27-2031 (Supply of Hot Water)",
        snippet="The hot water minimum is section 27-2031",
        kind="DOC",
    )
    return citations


def _context(citations: CitationRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        deps=SimpleNamespace(
            citations=citations,
            query=QUERY,
            user_history=QUERY,
            validation_rejections=[],
        ),
        retry=0,
        max_retries=2,
        loaded_capability_ids={"housing-hot-water-code-section"},
        capabilities={},
    )


def _section_tool() -> Tool:
    async def handler(args, ctx):
        citation_id = ctx.citations.register(
            "https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-236495",
            title="NYC Housing Maintenance Code section 27-2031 (Supply of Hot Water)",
            snippet="Section 27-2031 covers hot water. File a complaint by calling 311.",
            kind="DOC",
        )
        return f"Section 27-2031 covers hot water. Call 311. {{cite:{citation_id}}}"

    return Tool(
        name="housing_guidance",
        description="Get official housing guidance",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        module="housing",
    )


def _section_registry() -> Registry:
    return Registry([
        ServiceModule(
            name="housing",
            description="Help with NYC housing problems",
            prompt="Use housing guidance and cite it.",
            situations=[
                SituationHint(
                    name="hot_water_code_section",
                    definition=(
                        "The resident explicitly asks which NYC code section covers a "
                        "hot-water outage, including a short follow-up in any language."
                    ),
                    focus_tools=["housing_guidance"],
                )
            ],
        )
    ])


def test_hot_water_section_postcondition_requires_matching_citation():
    guard = getattr(agent, "_hot_water_section_feedback", None)
    assert callable(guard)
    citations = _citations().mapping()

    assert guard(True, "311-এ অভিযোগ করুন। {cite:S1}", citations)
    assert guard(True, "Section 27-2031 প্রযোজ্য। {cite:S2}", citations)
    assert guard(True, "Section 27-2031 প্রযোজ্য। {cite:S3}", citations) is None


def test_hot_water_section_postcondition_uses_semantic_signal_only():
    guard = getattr(agent, "_hot_water_section_feedback", None)
    assert callable(guard)

    assert guard(False, "Section 27-2031 applies. {cite:S3}", _citations().mapping()) is None


def test_hot_water_section_postcondition_requires_retrieval_before_answering():
    guard = getattr(agent, "_hot_water_section_feedback", None)
    assert callable(guard)

    feedback = guard(True, "311-এ অভিযোগ করুন।", {})
    assert "housing_guidance" in feedback
    assert "do not guess" in feedback


def test_hot_water_section_postcondition_splits_bengali_sentence_boundaries():
    guard = getattr(agent, "_hot_water_section_feedback", None)
    assert callable(guard)

    answer = "Section 27-2031 প্রযোজ্য। অন্য তথ্য। {cite:S3}"
    assert guard(True, answer, _citations().mapping())


@pytest.mark.asyncio
async def test_pydantic_validator_retries_omitted_hot_water_section():
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    citations = _citations()
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(text="গরম পানির সমস্যাটি 311-এ জানান", citation_ids=["S1"]),
        GroundedBlock(text="27-2029 শুধু গরম রাখার নিয়ম", citation_ids=["S2"]),
    ])

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(_context(citations), output)


@pytest.mark.asyncio
async def test_pydantic_validator_retries_when_section_source_was_not_retrieved():
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    citations = CitationRegistry()
    citations.register(
        "https://www.nyc.gov/site/hpd/services-and-information/heat-and-hot-water-information.page",
        title="Heat and Hot Water Information, NYC HPD",
        snippet="File a heat or hot-water complaint by calling 311",
        kind="DOC",
    )
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(text="311-এ অভিযোগ করুন", citation_ids=["S1"]),
    ])

    with pytest.raises(ModelRetry, match="housing_guidance"):
        await runtime._validate_grounding(_context(citations), output)


@pytest.mark.asyncio
async def test_pydantic_validator_uses_loaded_situation_for_short_bengali_followup():
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    citations = _citations()
    context = _context(citations)
    context.deps.query = "কোন section?"
    context.deps.user_history = f"{QUERY}\nকোন section?"
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(text="311-এ অভিযোগ করুন", citation_ids=["S1"]),
    ])

    with pytest.raises(ModelRetry):
        await runtime._validate_grounding(context, output)


@pytest.mark.asyncio
async def test_pydantic_validator_accepts_hot_water_section_with_its_source():
    runtime = object.__new__(PydanticRuntimeAdapter)
    runtime._semantic_verifier = None
    citations = _citations()
    output = GroundedAnswer(grounded_blocks=[
        GroundedBlock(text="গরম পানির সমস্যাটি 311-এ জানান", citation_ids=["S1"]),
        GroundedBlock(text="গরম পানির নিয়মটি section 27-2031-এ আছে", citation_ids=["S3"]),
    ])

    assert await runtime._validate_grounding(_context(citations), output) == output


@pytest.mark.asyncio
async def test_pydantic_runtime_recovers_after_required_section_retry():
    calls = 0

    async def model(messages, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([
                ToolCallPart(
                    "load_capability",
                    {"id": "housing-hot-water-code-section"},
                    "load-situation",
                )
            ])
        if calls == 2:
            return ModelResponse([
                ToolCallPart("load_capability", {"id": "housing"}, "load-housing")
            ])
        if calls == 3:
            return ModelResponse([
                ToolCallPart("housing_guidance", {}, "housing-guidance")
            ])
        output_name = info.output_tools[0].name
        if calls == 4:
            return ModelResponse([
                ToolCallPart(
                    output_name,
                    {
                        "grounded_blocks": [{
                            "text": "311-এ অভিযোগ করুন",
                            "citation_ids": ["S1"],
                        }]
                    },
                    "missing-section",
                )
            ])
        return ModelResponse([
            ToolCallPart(
                output_name,
                {
                    "grounded_blocks": [{
                        "text": "গরম পানির জন্য section 27-2031 প্রযোজ্য; 311-এ অভিযোগ করুন",
                        "citation_ids": ["S1"],
                    }]
                },
                "corrected-section",
            )
        ])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=_section_registry(),
        tools={"housing_guidance": _section_tool()},
        use_module_capabilities=True,
        structured_grounding=True,
    )

    result = await runtime.run(QUERY)

    assert calls == 5
    assert "27-2031" in result.text
    assert "{cite:S1}" in result.text
    assert result.diagnostics["validation_rejections"][0]["stage"] == "required_scope"


def test_ordinary_no_heat_answer_does_not_require_hot_water_section():
    guard = getattr(agent, "_hot_water_section_feedback", None)
    assert callable(guard)
    assert guard(
        False,
        "Call 311 to file a heat complaint. {cite:S1}",
        _citations().mapping(),
    ) is None


def test_housing_manifest_declares_hot_water_code_section_situation():
    hint = Registry.discover(Path("heynyc/modules")).situation_hints()[
        "hot_water_code_section"
    ][1]

    assert "explicitly asks" in hint.definition
    assert "housing_guidance" in hint.focus_tools


@pytest.mark.asyncio
async def test_legacy_last_iteration_abstains_instead_of_leaking_retry_prompt():
    async def complete(messages, tool_schemas):
        return {"role": "assistant", "content": "311-এ অভিযোগ করুন। {cite:S1}"}

    async def scope(user_message, history):
        return ScopeResult(
            model="injected",
            modules=("housing",),
            situations=("hot_water_code_section",),
        )

    runtime = Agent(
        Registry.discover(Path("heynyc/modules")),
        tools={},
        complete_fn=complete,
        scope_fn=scope,
    )
    result = await runtime.run(QUERY, max_iters=1)

    assert result.text == GROUNDING_ABSTAIN_FALLBACK
    assert "State section 27-2031" not in result.text
    assert result.messages[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_legacy_runtime_recovers_after_required_section_retry():
    calls = 0

    async def complete(messages, tool_schemas):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "housing-guidance",
                    "function": {"name": "housing_guidance", "arguments": "{}"},
                }],
            }
        if calls == 2:
            return {"role": "assistant", "content": "311-এ অভিযোগ করুন। {cite:S1}"}
        return {
            "role": "assistant",
            "content": "গরম পানির জন্য section 27-2031 প্রযোজ্য। {cite:S1}",
        }

    async def scope(user_message, history):
        return ScopeResult(
            model="injected",
            modules=("housing",),
            situations=("hot_water_code_section",),
        )

    runtime = Agent(
        _section_registry(),
        tools={"housing_guidance": _section_tool()},
        complete_fn=complete,
        scope_fn=scope,
    )
    result = await runtime.run(QUERY, max_iters=4)

    assert calls == 3
    assert "27-2031" in result.text
    assert "{cite:S1}" in result.text
    assert result.status == "success"
