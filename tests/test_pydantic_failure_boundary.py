from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from heynyc.channels.format import render
from heynyc.core.citations import CitationRegistry
from heynyc.core.localization import localize
from heynyc.core.manifest import ServiceModule, SituationHint
from heynyc.core.pydantic_runtime import PydanticRuntimeAdapter
from heynyc.core.pydantic_runtime.projection import _resident_history
from heynyc.core.pydantic_runtime.runtime import (
    TEMPORARY_FAILURE_FALLBACK,
    UNVERIFIED_DRAFT_NOTICE,
    PydanticRunFailure,
    _current_turn_citation_ids,
    _degraded_failure_text,
    _validation_citation_ids,
    _validation_warning_text,
)
from heynyc.core.registry import Registry
from heynyc.core.tools.base import Tool, ToolContext, ToolFailureError


def _cited_answer(answer: str, call_id: str = "answer-1") -> ToolCallPart:
    return ToolCallPart(
        "final_answer",
        {"answer": answer},
        call_id,
    )


async def test_native_runtime_returns_expected_tool_failures_to_the_model() -> None:
    async def unavailable(_args, _ctx):
        raise ToolFailureError(
            status="unavailable",
            reason="The source blocked retrieval.",
            retryable=False,
            source_url="https://example.org/blocked",
        )

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            return ModelResponse([ToolCallPart("unavailable", {}, "tool-1")])
        assert "The source blocked retrieval." in str(tool_returns[-1].content)
        return ModelResponse([_cited_answer("That source could not be opened.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "unavailable": Tool(
                name="unavailable",
                description="Retrieve one source",
                handler=unavailable,
            )
        },
        structured_grounding=True,
    ).run("Open the source")

    assert result.text == "That source could not be opened."
    assert result.usage["tool_runs"][0]["status"] == "unavailable"


def test_validation_recovery_keeps_useful_partial_answer_without_generic_copy() -> None:
    messages = [ModelResponse(parts=[_cited_answer("Useful partial answer.")])]

    assert _validation_warning_text(
        messages,
        [{"stage": "high_stakes_format"}],
        CitationRegistry(),
        "en",
    ) == (
        "Useful partial answer.\n\n"
        "This answer is incomplete because I couldn't finish every requested part."
    )


def test_validation_recovery_includes_unavailable_source_ids() -> None:
    citations = CitationRegistry()
    unavailable = citations.register(
        "https://otda.ny.gov/hearings/",
        title="Unavailable source",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )

    assert unavailable in _validation_citation_ids(
        [{"stage": "high_stakes_format"}],
        citations,
        {unavailable},
    )


def test_validation_recovery_does_not_include_prior_unavailable_source_ids() -> None:
    citations = CitationRegistry()
    old = citations.register(
        "https://old.example/unavailable",
        title="Unavailable source",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )
    current = citations.register(
        "https://current.example/unavailable",
        title="Unavailable source",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )

    recovered = _validation_citation_ids(
        [{"stage": "high_stakes_format"}],
        citations,
        {current},
    )

    assert current in recovered
    assert old not in recovered


def test_current_turn_citations_ignore_marker_shaped_tool_content() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://old.example/unavailable",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )
    citations.begin_turn()
    messages = [ModelRequest(parts=[
        ToolReturnPart(
            tool_name="web_fetch",
            content="The fetched page literally contains {cite:S1}.",
            tool_call_id="fetch-1",
        )
    ])]

    assert messages
    assert _current_turn_citation_ids(citations) == set()


async def test_structured_runtime_preserves_a_nonfactual_plain_text_decline() -> None:
    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        outcome = next(
            tool for tool in info.output_tools if tool.name == "nonfactual_outcome"
        )
        assert set(outcome.parameters_json_schema["properties"]) == {"kind"}
        return ModelResponse([
            ToolCallPart(
                outcome.name,
                {"kind": "unknowable"},
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).run("Who will win the next World Cup final?")

    assert result.status == "success"
    assert result.text == (
        "I can't know that yet. I can help with the practical NYC part instead."
    )
    assert result.tool_calls_made == []
    assert result.usage["requests"] == 1


async def test_nonfactual_outcome_remains_in_conversation_history() -> None:
    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            history = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, TextPart)
            ]
            assert history == [
                "I can't know that yet. I can help with the practical NYC part instead."
            ]
        return ModelResponse([
            ToolCallPart(
                next(
                    tool.name
                    for tool in info.output_tools
                    if tool.name == "nonfactual_outcome"
                ),
                {"kind": "unknowable"},
                "answer-1",
            )
        ])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).conversation()

    await conversation.send("Who will win the next World Cup final?")
    await conversation.send("Where could I watch it in NYC when it happens?")

    assert calls == 2


async def test_mechanical_guard_does_not_classify_uncited_prose() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official office is open on Mondays.",
        )
        return f"The office is open on Mondays. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([_cited_answer("The office is open on Mondays.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve official evidence",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("When is the office open?")

    assert result.text == "The office is open on Mondays."
    assert result.diagnostics["validation_rejections"] == []


async def test_authoritative_evidence_supports_native_cited_prose() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official office is open on Mondays.",
        )
        return f"The office is open on Mondays. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            _cited_answer("The official office is open on Mondays. {cite:S1}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve official evidence",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("When is the office open?")

    assert result.status == "success"
    assert result.text.startswith("The official office is open on Mondays.")


async def test_successful_answer_omits_an_unused_unavailable_source_url() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://otda.ny.gov/hearings/",
            title="Unavailable source",
            kind="WEB",
            snippet="No page content was retrieved.",
            provenance={"evidence_grade": "unavailable"},
        )
        return f"The page could not be fetched. {{cite:{citation_id}}}"

    calls = 0

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([_cited_answer("Here is the useful partial answer.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Try an official source",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("Help")

    assert result.status == "success"
    assert "Here is the useful partial answer." in result.text
    assert "https://otda.ny.gov/hearings/" not in result.text
    assert "Unverified source" not in result.text


async def test_later_turn_does_not_show_an_unrelated_unavailable_source() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://otda.ny.gov/hearings/",
            title="Unavailable source",
            kind="WEB",
            snippet="No page content was retrieved.",
            provenance={"evidence_grade": "unavailable"},
        )
        return f"The page could not be fetched. {{cite:{citation_id}}}"

    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        answer = "First partial answer." if calls == 2 else "Unrelated second answer."
        return ModelResponse([_cited_answer(answer, f"answer-{calls}")])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Try an official source",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).conversation()

    first = await conversation.send("First topic")
    second = await conversation.send("Unrelated second topic")

    assert "https://otda.ny.gov/hearings/" not in first.text
    assert second.text == "Unrelated second answer."


async def test_later_turn_omits_an_unused_unavailable_source_retried_this_turn() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://otda.ny.gov/hearings/",
            title="Unavailable source",
            kind="WEB",
            snippet="No page content was retrieved.",
            provenance={"evidence_grade": "unavailable"},
        )
        return f"The page could not be fetched. {{cite:{citation_id}}}"

    calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return ModelResponse([
                ToolCallPart("retrieve", {}, f"retrieve-{calls}")
            ])
        return ModelResponse([
            _cited_answer(f"Partial answer {calls // 2}.", f"answer-{calls}")
        ])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Try an official source",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).conversation()

    first = await conversation.send("First attempt")
    second = await conversation.send("Try that source again")

    assert "https://otda.ny.gov/hearings/" not in first.text
    assert "https://otda.ny.gov/hearings/" not in second.text


def test_recovery_source_title_cannot_inject_a_markdown_link() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://trusted.example/source",
        title="Official](<https://evil.example/phish>) [Page",
        snippet="No page content was retrieved.",
        provenance={"evidence_grade": "unavailable"},
    )
    result = SimpleNamespace(
        text=_degraded_failure_text("Useful partial answer.", citations),
        citations=citations.mapping(),
        diagnostics={},
        action_links=(),
    )

    rendered = "\n".join(render(result, "sms_twilio"))

    assert "https://trusted.example/source" in rendered
    assert "https://evil.example/phish" not in rendered


def test_recovery_shows_one_unavailable_page_per_site() -> None:
    citations = CitationRegistry()
    for url in (
        "https://otda.ny.gov/hearings/",
        "https://otda.ny.gov/hearings/faq.asp",
        "https://otda.ny.gov/hearings/request/",
        "https://www.nyc.gov/site/hra/help.page",
    ):
        citations.register(
            url,
            title="Unavailable source",
            snippet="No page content was retrieved.",
            provenance={"evidence_grade": "unavailable"},
        )

    text = _degraded_failure_text("Useful partial answer.", citations)
    result = SimpleNamespace(
        text=text,
        citations=citations.mapping(),
        diagnostics={},
        action_links=(),
    )
    rendered = "\n".join(render(result, "sms_twilio"))

    assert "https://otda.ny.gov/hearings/request/" in rendered
    assert "https://otda.ny.gov/hearings/faq.asp" not in rendered
    assert "https://otda.ny.gov/hearings/" not in rendered.replace(
        "https://otda.ny.gov/hearings/request/", ""
    )
    assert "https://www.nyc.gov/site/hra/help.page" in rendered


async def test_grounded_answer_does_not_need_completion_metadata() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="DATA",
            snippet="No current locations are confirmed open.",
        )
        return f"No current locations are confirmed open. {{cite:{citation_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            ToolCallPart(
                "final_answer",
                {
                    "answer": "No current locations are confirmed open.",
                },
                "answer-1",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve official evidence",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("Is any location open now?")

    assert result.text == "No current locations are confirmed open."
    assert result.diagnostics["validation_rejections"] == []


async def test_mechanical_guard_does_not_classify_prose_before_retrieval() -> None:
    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([_cited_answer("The office is open on Mondays.")])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).run("When is the office open?")

    assert result.text == "The office is open on Mondays."


@pytest.mark.parametrize(
    "follow_up",
    [
        "The office is open on Mondays.",
        "The office is open on Mondays; which time works for you?",
        "The office is open on Mondays?",
    ],
)
async def test_citation_free_clarification_cannot_include_factual_prose(
    follow_up: str,
) -> None:
    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse([
            ToolCallPart(
                "grounded_answer",
                {
                    "grounded_blocks": [],
                    "follow_up_question": follow_up,
                },
                "answer-1",
            )
        ])

    with pytest.raises(PydanticRunFailure) as raised:
        await PydanticRuntimeAdapter(
            FunctionModel(model),
            registry=Registry([]),
            tools={},
            structured_grounding=True,
        ).run("When is the office open?")

    assert raised.value.partial_result.status == "error"
    assert follow_up not in raised.value.partial_result.text
    assert raised.value.partial_result.text == TEMPORARY_FAILURE_FALLBACK
    assert raised.value.partial_result.diagnostics["failure_type"] == (
        "UnexpectedModelBehavior"
    )


async def test_accepted_structured_output_discards_sibling_plain_text() -> None:
    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert all(
                part.content != "The office is open on Mondays."
                for message in messages
                for part in message.parts
                if isinstance(part, TextPart)
            )
        return ModelResponse([
            TextPart("The office is open on Mondays."),
            ToolCallPart(
                next(
                    tool.name
                    for tool in info.output_tools
                    if tool.name == "nonfactual_outcome"
                ),
                {"kind": "unknowable"},
                f"answer-{calls}",
            ),
        ])

    conversation = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).conversation()

    first = await conversation.send("Who will win the next World Cup final?")
    second = await conversation.send("What about the one after that?")

    assert first.text == (
        "I can't know that yet. I can help with the practical NYC part instead."
    )
    assert second.text == first.text


def test_history_projection_discards_sibling_plain_text() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("Who will win?")]),
        ModelResponse(parts=[
            TextPart("The office is open on Mondays."),
            ToolCallPart(
                "nonfactual_outcome",
                {"kind": "unknowable"},
                "answer-1",
            ),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(
                "nonfactual_outcome",
                "Final result processed.",
                "answer-1",
            )
        ]),
    ]

    assert _resident_history(messages) == [
        {"role": "user", "content": "Who will win?"},
        {
            "role": "assistant",
            "content": (
                "I can't know that yet. "
                "I can help with the practical NYC part instead."
            ),
        },
    ]


async def test_mechanical_boundary_does_not_parse_phone_semantics() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/example",
            title="Official example",
            kind="WEB",
            snippet="The official source supports a different claim.",
        )
        return f"Official evidence. {{cite:{citation_id}}}"

    calls = 0

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            _cited_answer("Call the unsupported number 212-555-1212. {cite:S1}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve official evidence",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("Can you verify the number?")

    assert result.status == "success"
    assert result.text == "Call the unsupported number 212-555-1212. {cite:S1}"
    assert result.diagnostics["validation_rejections"] == []


async def test_exact_fact_guard_keeps_cited_document_evidence() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        clinic_id = ctx.citations.register(
            "https://data.cityofnewyork.us/example",
            title="Clinic row",
            kind="DATA",
            snippet="Apicha Community Health Center at 82-11 37th Ave.",
            provenance={"snapshot": {"name": "Apicha Community Health Center"}},
        )
        care_id = ctx.citations.register(
            "https://access.nyc.gov/programs/nyc-care/",
            title="NYC Care",
            kind="DOC",
            snippet="Enroll in NYC Care at 646-692-2273.",
        )
        return f"Clinic {{cite:{clinic_id}}}; enrollment {{cite:{care_id}}}"

    calls = 0

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            _cited_answer(
                "Apicha is one option. Call NYC Care at 646-692-2273. "
                "{cite:S1} {cite:S2}"
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "retrieve": Tool(
                name="retrieve",
                description="Retrieve clinic and maintained program evidence",
                handler=retrieve,
            )
        },
        structured_grounding=True,
    ).run("Where can I get care without insurance?")

    assert calls == 2
    assert result.status == "success"
    assert result.diagnostics["validation_rejections"] == []


async def test_exhausted_output_validation_is_not_returned_as_a_successful_fallback() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://data.cityofnewyork.us/example",
            title="Official service row",
            kind="DATA",
            snippet="Call 212-555-0100.",
            provenance={"snapshot": {"phone": "212-555-0100"}},
        )
        return f"Call 212-555-0100. {{cite:{citation_id}}}"

    calls = 0

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            _cited_answer(
                "Call 212-555-9999. {cite:S1}",
                f"answer-{calls}",
            )
        ])

    with pytest.raises(PydanticRunFailure) as raised:
        await PydanticRuntimeAdapter(
            FunctionModel(model),
            registry=Registry([]),
            tools={
                "retrieve": Tool(
                    name="retrieve",
                    description="Retrieve official evidence",
                    handler=retrieve,
                )
            },
            structured_grounding=True,
        ).run("What number should I call?")

    assert calls == 4
    assert raised.value.partial_result.status == "error"
    assert raised.value.partial_result.usage["tool_runs"][0]["tool"] == "retrieve"
    assert raised.value.partial_result.usage["tool_runs"][0]["status"] == "success"
    assert "212-555-9999" not in raised.value.partial_result.text
    assert "Call 212-555-0100." in raised.value.partial_result.text
    assert "Structured data record" not in raised.value.partial_result.text
    assert "Official service row" in raised.value.partial_result.text
    assert "Verified source" not in raised.value.partial_result.text
    assert "City data record" not in raised.value.partial_result.text
    assert raised.value.partial_result.text.rstrip().endswith(
        "I couldn't verify every detail in that answer. "
        "Check the linked sources before relying on it."
    )
    rendered = "\n".join(render(raised.value.partial_result, "sms_twilio"))
    assert "https://data.cityofnewyork.us/example" in rendered
    assert "311" not in raised.value.partial_result.text
    assert raised.value.partial_result.diagnostics["failure_type"] == (
        "UnexpectedModelBehavior"
    )
    assert raised.value.partial_result.diagnostics["validation_rejections"][-1][
        "stage"
    ] == "structured_grounding"


def test_recovery_source_urls_are_clickable_on_text_channels() -> None:
    citations = CitationRegistry()
    citations.register(
        "https://data.cityofnewyork.us/resource/erm2-nwe9.json?"
        "$where=unique_key='70056272' AND status='Closed'",
        title="NYC 311 request",
        kind="DATA",
        snippet="The request is closed.",
    )
    result = SimpleNamespace(
        text=_degraded_failure_text(UNVERIFIED_DRAFT_NOTICE, citations),
        citations=citations.mapping(),
        diagnostics={},
        action_links=(),
    )

    rendered = "\n".join(render(result, "sms_twilio"))

    url = rendered.split("https://data.cityofnewyork.us/", 1)[1].split("\n", 1)[0]
    assert "%24where" in url
    assert "%27" in url
    assert " " not in url


async def test_exhausted_grounding_keeps_supported_sibling_and_source_fact() -> None:
    async def retrieve(_args: dict, ctx: ToolContext) -> str:
        service_id = ctx.citations.register(
            "https://data.cityofnewyork.us/service",
            title="Official service row",
            kind="DATA",
            snippet="Official Service Center.",
            provenance={"snapshot": {"name": "Official Service Center"}},
        )
        phone_id = ctx.citations.register(
            "https://data.cityofnewyork.us/phone",
            title="Official phone row",
            kind="DATA",
            snippet="Call 212-555-0100.",
            provenance={"snapshot": {"phone": "212-555-0100"}},
        )
        return f"Service {{cite:{service_id}}}; phone {{cite:{phone_id}}}"

    async def model(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse([ToolCallPart("retrieve", {}, "retrieve-1")])
        return ModelResponse([
            _cited_answer(
                "Official Service Center. {cite:S1}\n\n"
                "Call 212-555-9999. {cite:S2}"
            )
        ])

    with pytest.raises(PydanticRunFailure) as raised:
        await PydanticRuntimeAdapter(
            FunctionModel(model),
            registry=Registry([]),
            tools={
                "retrieve": Tool(
                    name="retrieve",
                    description="Retrieve official evidence",
                    handler=retrieve,
                )
            },
            structured_grounding=True,
        ).run("What service and phone should I use?")

    text = raised.value.partial_result.text
    assert "Official Service Center. {cite:S1}" in text
    assert "212-555-9999" not in text
    assert "Call 212-555-0100." in text


async def test_high_stakes_authoritative_excerpt_is_preserved_when_exactly_supported() -> None:
    async def search(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/official-guidance",
            title="Search result",
            kind="WEB",
            snippet="The office is open on Mondays.",
            provenance={
                "evidence_grade": "authoritative_excerpt",
                "source_tier": "authoritative",
            },
        )
        return f"Search result. {{cite:{citation_id}}}"

    calls = 0
    retry_text = ""

    async def high_stakes_scope(_turns: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            model="test",
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            requests=0,
            cost_usd=0.0,
            latency_ms=0.0,
            modules=("benefits",),
            situations=("benefits_guidance",),
            event_turn=None,
        )

    async def model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls, retry_text
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("web_search", {}, "search-1")])
        if calls == 2:
            return ModelResponse([
                ToolCallPart(
                    "grounded_answer",
                    {"grounded_blocks": [{
                        "text": "The office is open on Mondays.",
                        "citation_ids": ["S1"],
                    }]},
                    "answer-1",
                )
            ])
        retry_text = "\n".join(
            str(getattr(part, "content", ""))
            for message in messages
            for part in message.parts
        )
        return ModelResponse([
            ToolCallPart(
                "grounded_answer",
                {"grounded_blocks": [{
                    "text": "The office is open on Mondays.",
                    "citation_ids": ["S1"],
                }]},
                "answer-2",
            )
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([
            ServiceModule(
                name="benefits",
                situations=[SituationHint(
                    name="benefits_guidance",
                    definition="Guidance that can affect a resident's benefits.",
                    high_stakes=True,
                )],
            )
        ]),
        tools={
            "web_search": Tool(
                name="web_search",
                description="Find a source",
                handler=search,
            ),
        },
        structured_grounding=True,
        scope_screen=high_stakes_scope,
    ).run("When is the office open?")

    assert result.text == "The office is open on Mondays. {cite:S1}"
    assert calls == 2
    assert retry_text == ""


async def test_low_stakes_discovery_excerpt_does_not_add_a_failure_notice() -> None:
    async def search(_args: dict, ctx: ToolContext) -> str:
        citation_id = ctx.citations.register(
            "https://www.nyc.gov/official-guidance",
            title="NYC office guidance",
            kind="WEB",
            snippet="The office is open on Mondays.",
            provenance={"evidence_grade": "discovery"},
        )
        return f"Search result. {{cite:{citation_id}}}"

    calls = 0

    async def model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart("search", {}, "search-1")])
        return ModelResponse([
            _cited_answer("The office is open on Mondays. {cite:S1}", f"answer-{calls}")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "search": Tool(
                name="search",
                description="Find a source",
                handler=search,
            )
        },
        structured_grounding=True,
    ).run("When is the office open?")

    assert calls == 2
    assert result.status == "success"
    assert result.text == "The office is open on Mondays. {cite:S1}"


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (
            "es",
            "Nota de verificación para {source}: esta fuente es un fragmento de un resultado "
            "de búsqueda. No pude confirmarla en la página completa.",
        ),
        (
            "bn",
            "{source}-এর জন্য যাচাইকরণ নোট: এই উৎসটি একটি অনুসন্ধান ফলাফলের অংশ। "
            "আমি সম্পূর্ণ পৃষ্ঠা থেকে এটি নিশ্চিত করতে পারিনি।",
        ),
        (
            "zh",
            "{source} 的核实说明：此来源只是搜索结果摘要，我无法从完整页面确认该信息。",
        ),
    ],
)
def test_discovery_validation_notice_is_localized(
    language: str,
    expected: str,
) -> None:
    assert localize(
        "Verification note for {source}: this source is a search-result excerpt. "
        "I could not confirm it from the full page.",
        language,
    ) == expected


def test_unverified_draft_notice_is_localized() -> None:
    assert localize(
        "I couldn't verify every detail in that answer. "
        "Check the linked sources before relying on it.",
        "es",
    ) == (
        "No pude verificar todos los detalles de esa respuesta. "
        "Revisa las fuentes enlazadas antes de confiar en ella."
    )


def test_claim_specific_notice_is_localized() -> None:
    assert localize(
        "I couldn't confirm this from the sources I checked:",
        "es",
    ) == "No pude confirmar lo siguiente con las fuentes que consulté:"


async def test_rejected_final_answer_cannot_switch_to_a_clarification() -> None:
    calls = 0

    async def model(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([
                _cited_answer("This uses an unknown source. {cite:S999}", "answer-1")
            ])
        assert {tool.name for tool in info.output_tools} == {"final_answer"}
        if calls == 2:
            return ModelResponse([
                ToolCallPart(
                    "clarification_request",
                    {"question": "Can you clarify?"},
                    "clarification-1",
                )
            ])
        return ModelResponse([
            _cited_answer("I could not establish that from the available evidence.", "answer-2")
        ])

    result = await PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={},
        structured_grounding=True,
    ).run("Can you verify this?")

    assert calls == 3
    assert result.status == "success"
    assert result.text == "I could not establish that from the available evidence."
    assert result.diagnostics["validation_rejections"] == [
        {"attempt": 1, "stage": "unknown_citation"}
    ]
