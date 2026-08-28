import pytest
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from heynyc.core.grounding import _cited_claims
from heynyc.core.pydantic_runtime.projection import (
    GroundedAnswer,
    GroundedBlock,
    _render_grounded_answer,
    _resident_history,
)


def test_pure_framing_does_not_require_a_source() -> None:
    block = GroundedBlock(kind="framing", text="I’m sorry you’re dealing with this.")

    assert block.citation_ids == []


def test_privacy_safe_follow_up_does_not_require_a_source() -> None:
    block = GroundedBlock(
        kind="question",
        text="Which neighborhood or nearby landmark works for you? Do not share a case number.",
    )

    assert block.citation_ids == []


def test_output_schema_prioritizes_urgent_help_and_keeps_empathy_factual_free() -> None:
    schema = GroundedAnswer.model_json_schema()

    assert "urgent" in schema["properties"]["grounded_blocks"]["description"].lower()
    assert "assistant follow-up" in schema["properties"]["grounded_blocks"][
        "description"
    ].lower()
    assert "kind=question" in schema["properties"]["grounded_blocks"][
        "description"
    ].lower()
    assert "must not restate" in schema["$defs"]["GroundedBlock"]["properties"]["kind"][
        "description"
    ].lower()


def test_claim_still_requires_a_source() -> None:
    with pytest.raises(ValidationError):
        GroundedBlock(kind="claim", text="The office is open today.")


def test_question_cannot_declare_a_source() -> None:
    with pytest.raises(ValidationError):
        GroundedBlock(
            kind="question",
            text="Which neighborhood works for you?",
            citation_ids=["S1"],
        )


def test_framing_cannot_hide_a_resident_visible_source() -> None:
    with pytest.raises(ValidationError):
        GroundedBlock(
            kind="framing",
            text="That nearby search may not include every report from the building.",
            citation_ids=["S1"],
        )


def test_history_migrates_accepted_legacy_framing_citations() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("What can I do?")]),
        ModelResponse(parts=[ToolCallPart(
            "grounded_answer",
            {
                "grounded_blocks": [
                    {
                        "kind": "claim",
                        "text": "Use the official application.",
                        "citation_ids": ["S1"],
                    },
                    {
                        "kind": "framing",
                        "text": "I could not verify the deadline.",
                        "citation_ids": ["S1"],
                    },
                ]
            },
            "accepted",
        )]),
        ModelRequest(parts=[ToolReturnPart(
            "grounded_answer",
            "Final result processed.",
            "accepted",
        )]),
    ]

    assert _resident_history(messages) == [
        {"role": "user", "content": "What can I do?"},
        {
            "role": "assistant",
            "content": (
                "Use the official application. {cite:S1} "
                "I could not verify the deadline."
            ),
        },
    ]


@pytest.mark.parametrize(
    "citation_ids",
    ["S1", 1, {"S1": True}, [123]],
)
def test_history_rejects_malformed_legacy_framing_citations(citation_ids) -> None:
    messages = [
        ModelResponse(parts=[ToolCallPart(
            "grounded_answer",
            {
                "grounded_blocks": [{
                    "kind": "framing",
                    "text": "I could not verify the deadline.",
                    "citation_ids": citation_ids,
                }]
            },
            "accepted",
        )]),
        ModelRequest(parts=[ToolReturnPart(
            "grounded_answer",
            "Final result processed.",
            "accepted",
        )]),
    ]

    with pytest.raises(ValidationError):
        _resident_history(messages)


def test_same_paragraph_claims_keep_their_own_citations() -> None:
    answer = GroundedAnswer(grounded_blocks=[
        GroundedBlock(
            text="The bounded search returned 10 nearby rodent complaints.",
            citation_ids=["S1"],
        ),
        GroundedBlock(
            text="The newest complaint was opened August 13, 2026.",
            citation_ids=["S2"],
        ),
    ])

    assert [
        (claim, citation_ids)
        for _original, claim, citation_ids in _cited_claims(
            _render_grounded_answer(answer)
        )
    ] == [
        ("The bounded search returned 10 nearby rodent complaints.", ["S1"]),
        ("The newest complaint was opened August 13, 2026.", ["S2"]),
    ]


def test_explicit_paragraph_break_preserves_long_answer_structure() -> None:
    answer = GroundedAnswer(grounded_blocks=[
        GroundedBlock(
            kind="claim",
            text="The first option is open today.",
            citation_ids=["S1"],
        ),
        GroundedBlock(
            kind="claim",
            text="The second option is in Queens.",
            citation_ids=["S2"],
            starts_new_paragraph=True,
        ),
    ])

    assert _render_grounded_answer(answer) == (
        "The first option is open today. {cite:S1}\n\n"
        "The second option is in Queens. {cite:S2}"
    )


def test_omitted_paragraph_metadata_preserves_list_items() -> None:
    answer = GroundedAnswer(grounded_blocks=[
        GroundedBlock(
            kind="framing",
            text="Here are two options:",
        ),
        GroundedBlock(
            kind="claim",
            text="- The first option is open today.",
            citation_ids=["S1"],
        ),
        GroundedBlock(
            kind="claim",
            text="- The second option is in Queens.",
            citation_ids=["S2"],
        ),
    ])

    assert _render_grounded_answer(answer) == (
        "Here are two options:\n"
        "- The first option is open today. {cite:S1}\n"
        "- The second option is in Queens. {cite:S2}"
    )
