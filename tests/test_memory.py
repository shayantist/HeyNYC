from __future__ import annotations

import pytest

from heynyc.core.agent import Agent, _history_messages, _routing_query
from heynyc.core.memory import (
    ContextCapacityError,
    ContinuityRecord,
    prepare_context,
)


def _turns(*labels: str) -> list[dict]:
    turns = []
    for label in labels:
        turns.extend([
            {"role": "user", "content": f"user {label}"},
            {"role": "assistant", "content": f"assistant {label}"},
        ])
    return turns


def _measure(history: list[dict], continuity: ContinuityRecord | None) -> int:
    continuity_size = 0
    if continuity:
        continuity_size = sum(
            len(value) if isinstance(value, str) else sum(len(item) for item in value)
            for value in continuity.model_dump().values()
        )
    return sum(len(str(turn.get("content") or "")) for turn in history) + continuity_size


@pytest.mark.asyncio
async def test_compaction_runs_only_when_measured_budget_is_exceeded():
    calls = []

    async def compact(older, current):
        calls.append(older)
        return ContinuityRecord(goal="user one")

    roomy = await prepare_context(
        _turns("one", "two"), None, budget=1_000, measure=_measure, compact=compact,
    )
    assert roomy.history == _turns("one", "two")
    assert not roomy.compacted
    assert calls == []

    pressured = await prepare_context(
        _turns("one", "two", "three"), None, budget=50, measure=_measure, compact=compact,
    )
    assert pressured.compacted
    assert calls == [_turns("one", "two")]
    assert pressured.history == _turns("three")
    assert pressured.continuity.goal == "user one"


@pytest.mark.asyncio
async def test_context_budget_keeps_only_complete_recent_turns():
    async def compact(older, current):
        return ContinuityRecord(goal="user one")

    plan = await prepare_context(
        _turns("one", "two", "three"), None, budget=60, measure=_measure, compact=compact,
    )

    assert len(plan.history) % 2 == 0
    assert plan.history[0]["role"] == "user"
    assert plan.history[-1]["role"] == "assistant"
    assert plan.history == _turns("two", "three")


@pytest.mark.asyncio
async def test_invalid_continuity_schema_fails_closed():
    async def compact(older, current):
        return {"goal": "continue", "official_deadline": "tomorrow"}

    with pytest.raises(ContextCapacityError, match="continuity"):
        await prepare_context(
            _turns("one", "two"), None, budget=30, measure=_measure, compact=compact,
        )


@pytest.mark.asyncio
async def test_unknown_context_capacity_fails_before_compaction():
    called = False

    async def compact(older, current):
        nonlocal called
        called = True
        return ContinuityRecord()

    with pytest.raises(ContextCapacityError, match="capacity"):
        await prepare_context(
            _turns("one"), None, budget=None, measure=_measure, compact=compact,
        )
    assert not called


@pytest.mark.asyncio
async def test_compaction_cannot_invent_user_facts_or_corrections():
    async def compact(older, current):
        return ContinuityRecord(
            user_facts=["I live in Queens"],
            corrections=["I never said I live in Brooklyn"],
        )

    with pytest.raises(ContextCapacityError, match="resident-authored"):
        await prepare_context(
            _turns("I live in Queens", "two"), None,
            budget=50, measure=_measure, compact=compact,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal", "Apply for SNAP"),
        ("completed_steps", ["The screener was submitted"]),
        ("unresolved_questions", ["Whether the resident qualifies"]),
    ],
)
async def test_compaction_cannot_invent_any_continuity_field(field, value):
    async def compact(older, current):
        return ContinuityRecord(**{field: value})

    with pytest.raises(ContextCapacityError, match="resident-authored"):
        await prepare_context(
            _turns("I need food help", "two"), None,
            budget=40, measure=_measure, compact=compact,
        )


def test_prior_assistant_turns_are_labeled_historical_context():
    """F052 (RULED 2026-07-17): prior assistant turns stay readable for continuity, labeled as
    possibly stale, with citation markers stripped; the unknown-citation and grounding guards
    are the deterministic stale-evidence backstop."""
    history = [{
        "role": "assistant",
        "content": "The office closes at 5 p.m. https://example.gov/old {cite:S1}",
    }]

    messages = _history_messages(history)

    assert "5 p.m." in messages[0]["content"]
    assert "{cite:" not in messages[0]["content"]
    assert "may be stale" in messages[0]["content"].lower()
    assert "retrieve current evidence" in messages[0]["content"].lower()
    # F062: the label must also mark the turn as SHARED context the resident already read, so
    # follow-ups build on it instead of re-introducing it ("You're probably asking about...").
    assert "shared context" in messages[0]["content"].lower()
    assert "re-introduce" in messages[0]["content"].lower()


def test_stamped_prior_turns_carry_their_sent_time_instead_of_blanket_staleness():
    """F062 (owner, 2026-07-18): a reply from a minute ago is not 'stale' like one from last
    week. When a turn carries its `timestamp`, the label states WHEN it was sent and the
    model weighs it against the current date line — no blanket may-be-stale framing."""
    history = [{
        "role": "assistant",
        "content": "The office closes at 5 p.m. {cite:S1}",
        "timestamp": "2026-07-15T09:30:00-04:00",
    }]

    messages = _history_messages(history)

    content = messages[0]["content"]
    assert "sent 2026-07-15 09:30 ET" in content
    assert "shared context" in content.lower()
    assert "retrieve current evidence" in content.lower()
    assert "may be stale" not in content.lower()


def test_prior_assistant_facts_are_not_used_to_build_the_system_prompt():
    query = _routing_query(
        "what about tomorrow?",
        [
            {"role": "user", "content": "When is the office open?"},
            {"role": "assistant", "content": "It closes at 5 p.m. https://example.gov/old"},
        ],
    )

    assert "When is the office open?" in query
    assert "5 p.m." not in query
    assert "https://" not in query


@pytest.mark.asyncio
async def test_malformed_compaction_still_accrues_spend(monkeypatch):
    class Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class Message:
        parsed = None
        content = "not valid json"

    class Response:
        usage = Usage()
        choices = [type("Choice", (), {"message": Message()})()]

    async def acompletion(**kwargs):
        return Response()

    monkeypatch.setattr("litellm.acompletion", acompletion)
    monkeypatch.setattr(
        "heynyc.core.agent.priced_cost_usd", lambda model, input_tokens, output_tokens: 0.01,
    )
    agent = Agent.__new__(Agent)
    agent._spend = type("Spend", (), {
        "recorded": [],
        "halt_reason": lambda self: None,
        "record": lambda self, model, input_tokens, output_tokens: self.recorded.append(
            (model, input_tokens, output_tokens)
        ),
        "mark_unpriceable": lambda self: None,
    })()

    with pytest.raises(Exception):
        await agent._compact_memory([], None)

    assert agent._spend.recorded == [("openai/gpt-5.4-nano", 11, 7)]


@pytest.mark.asyncio
async def test_empty_compaction_response_still_accrues_spend(monkeypatch):
    class Usage:
        prompt_tokens = 13
        completion_tokens = 0

    class Response:
        usage = Usage()
        choices = []

    async def acompletion(**kwargs):
        return Response()

    monkeypatch.setattr("litellm.acompletion", acompletion)
    monkeypatch.setattr(
        "heynyc.core.agent.priced_cost_usd", lambda model, input_tokens, output_tokens: 0.01,
    )
    agent = Agent.__new__(Agent)
    agent._spend = type("Spend", (), {
        "recorded": [],
        "halt_reason": lambda self: None,
        "record": lambda self, model, input_tokens, output_tokens: self.recorded.append(
            (model, input_tokens, output_tokens)
        ),
        "mark_unpriceable": lambda self: None,
    })()

    with pytest.raises(Exception):
        await agent._compact_memory([], None)

    assert agent._spend.recorded == [("openai/gpt-5.4-nano", 13, 0)]


@pytest.mark.asyncio
async def test_existing_continuity_is_revalidated_against_resident_history():
    current = ContinuityRecord(goal="invented official deadline")

    async def compact(older, previous):
        return previous

    with pytest.raises(ContextCapacityError, match="resident-authored"):
        await prepare_context(
            _turns("new", "latest"), current,
            budget=1_000, measure=_measure, compact=compact,
        )


@pytest.mark.asyncio
async def test_continuity_rejects_sensitive_identifiers():
    async def compact(older, current):
        return ContinuityRecord(exact_user_excerpts=["SSN 123-45-6789"])

    with pytest.raises(ContextCapacityError, match="sensitive"):
        await prepare_context(
            [
                {"role": "user", "content": "SSN 123-45-6789"},
                {"role": "assistant", "content": "Do not send that here"},
                *_turns("latest"),
            ],
            None, budget=35, measure=_measure, compact=compact,
        )
