from heynyc.core.grounding import _cited_claims
from heynyc.core.pydantic_runtime.projection import (
    GroundedAnswer,
    GroundedBlock,
    _render_grounded_answer,
)


def test_framing_uses_evidence_without_claiming_inline_ownership() -> None:
    answer = GroundedAnswer(grounded_blocks=[
        GroundedBlock(
            kind="claim",
            text="The bounded search returned 10 nearby rodent complaints.",
            citation_ids=["S1"],
        ),
        GroundedBlock(
            kind="framing",
            text="That nearby search may not include every report from the building.",
            citation_ids=["S1"],
        ),
    ])

    assert _render_grounded_answer(answer) == (
        "The bounded search returned 10 nearby rodent complaints. {cite:S1} "
        "That nearby search may not include every report from the building."
    )


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
            citation_ids=["S1"],
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
