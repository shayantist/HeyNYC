import base64
import json

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from heynyc.core.pydantic_runtime.runtime import (
    ConversationState,
    PydanticAgentSession,
    PydanticRuntimeAdapter,
)
from heynyc.core.registry import Registry
from heynyc.core.session import Session, _decode_line, _encode_line
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import GeoPoint


def _runtime(model=None):
    return PydanticRuntimeAdapter(
        model or FunctionModel(lambda _messages, _info: ModelResponse([TextPart("done")])),
        registry=Registry([]),
        tools={},
        guard_grounding=False,
    )


def test_conversation_state_is_strict_versioned_and_loadable():
    runtime = _runtime()
    session = runtime.conversation()

    assert isinstance(session, PydanticAgentSession)
    assert isinstance(session.state, ConversationState)

    payload = json.loads(session.dump_state())
    assert payload["schema_version"] == 1
    assert payload["conversation_id"]

    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        runtime.conversation_from_state(json.dumps(payload).encode())


def test_conversation_state_migrates_version_zero_and_rejects_future_versions():
    runtime = _runtime()
    payload = json.loads(runtime.conversation().dump_state())
    citations = payload.pop("citations")
    payload.pop("schema_version")
    payload.pop("conversation_id")
    payload["pending_citations"] = citations

    restored = runtime.conversation_from_state(json.dumps(payload).encode())
    migrated = json.loads(restored.dump_state())

    assert migrated["schema_version"] == 1
    assert migrated["conversation_id"]
    assert migrated["citations"] == citations
    assert "pending_citations" not in migrated

    migrated["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported conversation state version"):
        runtime.conversation_from_state(json.dumps(migrated).encode())


def test_current_location_round_trips_with_complete_provider_payload():
    runtime = _runtime()
    session = runtime.conversation()
    session.state.current_location = GeoPoint(
        40.756031,
        -73.828535,
        "Main Street, Flushing, Queens, New York 11355",
        match_type="nominatim",
        resident_query="Main Street, Flushing",
        provider_id="123",
        provider_payload={"place_id": 123, "namedetails": {"name": "Main Street"}},
    )

    restored = runtime.conversation_from_state(session.dump_state())

    assert restored.state.current_location == session.state.current_location
    assert restored.state.current_location.provider_payload["place_id"] == 123


async def test_tool_context_location_persists_between_native_turns():
    seen: list[GeoPoint | None] = []

    async def remember(_args: dict, ctx: ToolContext) -> str:
        seen.append(ctx.current_location)
        if ctx.current_location is None:
            ctx.current_location = GeoPoint(
                40.756031,
                -73.828535,
                "Main Street, Flushing, Queens, New York 11355",
                resident_query="Main Street, Flushing",
            )
        return "remembered"

    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return ModelResponse([
                ToolCallPart("remember", {}, f"remember-{calls}")
            ])
        return ModelResponse([TextPart("done")])

    runtime = PydanticRuntimeAdapter(
        FunctionModel(model),
        registry=Registry([]),
        tools={
            "remember": Tool(
                name="remember",
                description="Remember one resolved location",
                parameters={"type": "object", "properties": {}},
                handler=remember,
            )
        },
        guard_grounding=False,
    )
    session = runtime.conversation()

    await session.send("I am on Main Street in Flushing")
    await session.send("I am still there")

    assert seen[0] is None
    assert seen[1] == session.state.current_location


async def test_runtime_passes_persisted_conversation_id_to_pydantic():
    seen = []

    async def model(messages, _info):
        seen.append(messages[-1].conversation_id)
        return ModelResponse([TextPart("done")])

    runtime = _runtime(FunctionModel(model))
    session = runtime.conversation()
    conversation_id = session.state.conversation_id

    await session.send("hello")
    restored = runtime.conversation_from_state(session.dump_state())
    await restored.send("hello again")

    assert seen == [conversation_id, conversation_id]


async def test_real_pydantic_state_uses_the_session_lifecycle(tmp_path):
    runtime = _runtime()
    path = tmp_path / "session.jsonl"
    session = Session(runtime, "anonymous-session", path)

    assert session.convo.state.conversation_id == "anonymous-session"

    pending = await session.prepare("hello")
    assert session.convo.state.user_turns == ()
    session.commit(pending)

    restored = Session.load(runtime, "anonymous-session", path)
    assert restored.convo.state.conversation_id == "anonymous-session"
    assert restored.convo.state.user_turns == ("hello",)

    restored.reset()
    assert restored.convo.state.conversation_id == "anonymous-session"
    assert restored.convo.state.user_turns == ()


async def test_version_zero_session_reload_reuses_the_known_session_id(tmp_path):
    runtime = _runtime()
    path = tmp_path / "legacy-session.jsonl"
    session = Session(runtime, "known-session", path)
    session.commit(await session.prepare("hello"))

    record = _decode_line(path.read_text().strip())
    state = json.loads(base64.b64decode(record["state"]))
    state.pop("schema_version")
    state.pop("conversation_id")
    record["state"] = base64.b64encode(json.dumps(state).encode()).decode()
    path.write_text(_encode_line(record) + "\n")

    restored = Session.load(runtime, "known-session", path)

    assert restored.convo.state.conversation_id == "known-session"


def test_transcript_only_reload_reuses_the_known_session_id(tmp_path):
    runtime = _runtime()
    path = tmp_path / "legacy-transcript.jsonl"
    path.write_text(
        _encode_line({"role": "user", "content": "hello"})
        + "\n"
        + _encode_line({"role": "assistant", "content": "done"})
        + "\n"
    )

    restored = Session.load(runtime, "known-session", path)

    assert restored.convo.state.conversation_id == "known-session"
