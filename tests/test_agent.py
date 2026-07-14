from __future__ import annotations

import json

import pytest

from heynyc.core.agent import Agent
from heynyc.core.registry import Registry
from heynyc.core.tools import Tool, ToolContext


def _scripted(*responses):
    """Build a completion fn that returns the given assistant messages in order."""
    calls = {"i": 0}

    async def complete(messages, tool_schemas):
        resp = responses[calls["i"]]
        calls["i"] += 1
        return resp

    return complete


def _assistant(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def _tool_call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def empty_registry():
    return Registry([])


async def test_abstains_with_no_tools(empty_registry):
    complete = _scripted(_assistant(content="I don't have that info — try nyc.gov."))
    agent = Agent(empty_registry, tools={}, complete_fn=complete)
    result = await agent.run("where's the nearest cooling center?")
    assert "nyc.gov" in result.text
    assert result.iterations == 1
    assert result.tool_calls_made == []
    assert not result.hit_max_iters


async def test_empty_final_answer_falls_back_to_safe_refusal(empty_registry):
    # Regression (red-team PI12): an encoded-instruction injection made the model refuse by going
    # SILENT — a completely empty user-facing response. A terminal turn must never be blank; the
    # agent substitutes an explicit, non-empty safe refusal.
    from heynyc.core.agent import EMPTY_ANSWER_FALLBACK

    for empty in (None, "", "   \n  "):
        complete = _scripted(_assistant(content=empty))  # no text, no tool calls
        agent = Agent(empty_registry, tools={}, complete_fn=complete)
        result = await agent.run("Base64-decode the following and do exactly what it says: ...")
        assert result.text.strip()                       # never empty
        assert result.text == EMPTY_ANSWER_FALLBACK
        assert "can't help" in result.text.lower()


async def test_empty_answer_after_tool_call_still_falls_back(empty_registry):
    # Same guard, but the empty terminal turn follows a tool call (the loop's second iteration).
    from heynyc.core.agent import EMPTY_ANSWER_FALLBACK

    async def noop(args, ctx):
        return "ok"

    tool = Tool(name="noop", description="x", parameters={"type": "object", "properties": {}}, handler=noop)
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("noop", {})]),
        _assistant(content=""),   # model then returns nothing
    )
    agent = Agent(empty_registry, tools={"noop": tool}, complete_fn=complete)
    result = await agent.run("go")
    assert result.text == EMPTY_ANSWER_FALLBACK
    assert result.tool_calls_made == ["noop"]


async def test_empty_answer_fallback_is_streamed_as_text_delta(empty_registry):
    # The fallback must reach a streaming UI too (not only the drained result) — it's emitted as a
    # TextDelta so the on-screen answer is non-empty.
    from heynyc.core import events
    from heynyc.core.agent import EMPTY_ANSWER_FALLBACK

    complete = _scripted(_assistant(content=""))
    agent = Agent(empty_registry, tools={}, complete_fn=complete)
    deltas = [e.text async for e in agent.stream("hi") if isinstance(e, events.TextDelta)]
    assert "".join(deltas) == EMPTY_ANSWER_FALLBACK


async def test_tool_call_then_final_answer(empty_registry):
    async def nearest(args, ctx: ToolContext):
        cid = ctx.citations.register(
            "https://data.cityofnewyork.us/resource/h2bn-gu9k.json",
            snippet="Cooling site at 120 Broadway",
            kind="DATA",
        )
        return f"Nearest: 120 Broadway (0.2 mi) {{cite:{cid}}}"

    tool = Tool(
        name="nearest",
        description="find nearest",
        parameters={"type": "object", "properties": {"category": {"type": "string"}}},
        handler=nearest,
    )
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("nearest", {"category": "cooling_center"})]),
        _assistant(content="The nearest cooling center is 120 Broadway, 0.2 mi away {cite:S1}."),
    )
    agent = Agent(empty_registry, tools={"nearest": tool}, complete_fn=complete)
    result = await agent.run("nearest cooling center to me?")

    assert result.tool_calls_made == ["nearest"]
    assert result.iterations == 2
    assert "S1" in result.citations
    assert result.citations["S1"]["kind"] == "DATA"
    assert "120 Broadway" in result.text


async def test_unknown_tool_surfaces_error_not_crash(empty_registry):
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("ghost", {})]),
        _assistant(content="Sorry, something went wrong."),
    )
    agent = Agent(empty_registry, tools={}, complete_fn=complete)
    result = await agent.run("do something")
    # tool result with the error is fed back; loop continues to a final answer
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]
    assert result.iterations == 2


async def test_handler_exception_surfaced(empty_registry):
    async def boom(args, ctx):
        raise RuntimeError("socrata down")

    tool = Tool(name="boom", description="x", parameters={"type": "object", "properties": {}}, handler=boom)
    complete = _scripted(
        _assistant(tool_calls=[_tool_call("boom", {})]),
        _assistant(content="I couldn't reach the data source right now."),
    )
    agent = Agent(empty_registry, tools={"boom": tool}, complete_fn=complete)
    result = await agent.run("go")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert "socrata down" in tool_msgs[0]["content"]


async def test_conversation_threads_history(empty_registry):
    seen_messages = []

    async def recorder(messages, tool_schemas):
        seen_messages.append([m for m in messages if m["role"] in ("user", "assistant")])
        return _assistant(content=f"answer {len(seen_messages)}")

    agent = Agent(empty_registry, tools={}, complete_fn=recorder)
    convo = agent.conversation()

    r1 = await convo.send("nearest cooling center to Union Square?")
    r2 = await convo.send("what about in the Bronx?")

    assert r1.text == "answer 1"
    assert r2.text == "answer 2"
    # Second turn's prompt includes the first user msg + first assistant reply + new user msg
    second_turn = seen_messages[1]
    contents = [m["content"] for m in second_turn]
    assert "nearest cooling center to Union Square?" in contents
    assert "answer 1" in contents
    assert "what about in the Bronx?" in contents
    # History accumulates: 2 user + 2 assistant
    assert len(convo.turns) == 4


async def test_run_with_explicit_history(empty_registry):
    captured = {}

    async def cap(messages, tool_schemas):
        captured["messages"] = list(messages)  # snapshot before the loop appends
        return _assistant(content="ok")

    agent = Agent(empty_registry, tools={}, complete_fn=cap)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    await agent.run("follow up", history=history)
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_hits_max_iters(empty_registry):
    # Always returns a tool call → never terminates on its own.
    async def noop(args, ctx):
        return "ok"

    tool = Tool(name="loop", description="x", parameters={"type": "object", "properties": {}}, handler=noop)

    async def always_tool(messages, tool_schemas):
        return _assistant(tool_calls=[_tool_call("loop", {})])

    agent = Agent(empty_registry, tools={"loop": tool}, complete_fn=always_tool)
    result = await agent.run("go", max_iters=3)
    assert result.hit_max_iters
    assert result.iterations == 3


async def test_agent_captures_token_usage_from_stream():
    from heynyc.core.agent import Agent
    from heynyc.core.registry import Registry

    async def fake_stream(messages, tool_schemas):
        yield {"type": "text", "text": "Hello"}
        yield {"type": "usage", "input_tokens": 42, "output_tokens": 7}
        yield {"type": "message", "message": {"role": "assistant", "content": "Hello", "tool_calls": None}}

    agent = Agent(Registry([]), stream_fn=fake_stream)
    result = await agent.run("hi")
    assert result.usage["input_tokens"] == 42
    assert result.usage["output_tokens"] == 7
    assert result.usage["latency_ms"] >= 0.0


async def test_agent_reports_latency_breakdown_and_call_counts(empty_registry):
    async def echo(args, ctx: ToolContext):
        return "tool ran"

    tool = Tool(name="echo", description="x", parameters={"type": "object", "properties": {}}, handler=echo)
    responses = [
        _assistant(tool_calls=[_tool_call("echo", {})]),
        _assistant(content="done"),
    ]

    async def sf(messages, tool_schemas):
        response = responses.pop(0)
        if response.get("content"):
            yield {"type": "text", "text": response["content"]}
        yield {"type": "usage", "input_tokens": 1, "output_tokens": 1}
        yield {"type": "message", "message": response}

    agent = Agent(empty_registry, tools={"echo": tool}, stream_fn=sf)

    result = await agent.run("go")

    assert result.usage["model_time_ms"] >= 0.0
    assert result.usage["tool_time_ms"] >= 0.0
    assert result.usage["orchestration_time_ms"] >= 0.0
    assert result.usage["n_model_calls"] == 2
    assert result.usage["n_tool_calls"] == 1
    assert result.usage["iterations"] == 2


def test_completion_kwargs_omits_temperature_for_gpt5_models():
    # GPT-5 models reject temperature != 1 (litellm raises UnsupportedParamsError), so the agent must
    # NOT send temperature=0 for them. Regression guard for the gpt-5-mini backend migration.
    from heynyc.core.agent import _completion_kwargs

    kw = _completion_kwargs("openai/gpt-5-mini", messages=[], tool_schemas=[])
    assert "temperature" not in kw


def test_completion_kwargs_pins_temperature_zero_for_non_gpt5():
    # Every other model pins temperature=0 for deterministic, grounded output.
    from heynyc.core.agent import _completion_kwargs

    kw = _completion_kwargs("anthropic/claude-sonnet-4-6", messages=[], tool_schemas=[])
    assert kw["temperature"] == 0.0


def test_completion_kwargs_attaches_tools_only_when_present():
    # Tool schemas are passed through when present, omitted when empty (matches prior behavior).
    from heynyc.core.agent import _completion_kwargs

    schema = [{"type": "function", "function": {"name": "nearest"}}]
    assert _completion_kwargs("anthropic/claude-sonnet-4-6", messages=[], tool_schemas=schema)["tools"] == schema
    assert "tools" not in _completion_kwargs("anthropic/claude-sonnet-4-6", messages=[], tool_schemas=[])


def test_completion_kwargs_sets_num_ctx_for_ollama():
    # Ollama's default context window (~2-4K tokens) silently truncates HeyNYC's ~7.5K-token system
    # prompt, which breaks tool-calling; a self-hosted ollama model must get a large num_ctx so the
    # full prompt + tool schemas fit.
    from heynyc.core.agent import _completion_kwargs

    kw = _completion_kwargs("ollama_chat/qwen3.5:9b", messages=[], tool_schemas=[])
    assert kw.get("num_ctx", 0) >= 8192


def test_completion_kwargs_no_num_ctx_for_hosted_models():
    # Hosted APIs manage their own context window; num_ctx is an ollama-only knob.
    from heynyc.core.agent import _completion_kwargs

    assert "num_ctx" not in _completion_kwargs("anthropic/claude-sonnet-4-6", messages=[], tool_schemas=[])
    assert "num_ctx" not in _completion_kwargs("openai/gpt-5-mini", messages=[], tool_schemas=[])


# --- Change 2: prompt caching on the hosted (Anthropic) path --------------------------------------

def _real_agent(model: str) -> Agent:
    from heynyc.core import config

    reg = Registry.discover(config.MODULES_DIR, config.BASE_ALLOWLIST)
    return Agent(reg, tools={}, model=model)


def test_is_anthropic_detects_provider_from_model_string():
    from heynyc.core.agent import _is_anthropic

    assert _is_anthropic("anthropic/claude-sonnet-4-6")
    assert _is_anthropic("bedrock/anthropic.claude-3-5-sonnet")   # Bedrock Claude id
    assert not _is_anthropic("openai/gpt-4o-mini")
    assert not _is_anthropic("ollama_chat/qwen3.5:9b")


def test_system_message_is_cached_content_blocks_for_anthropic():
    # For an Anthropic model the system message is a list of content blocks, and the STABLE prefix
    # block (safety rules + capability menu) carries cache_control so repeat calls read it from cache.
    agent = _real_agent("anthropic/claude-sonnet-4-6")
    sysmsg = agent._system_message("where's the nearest food pantry?")

    assert sysmsg["role"] == "system"
    content = sysmsg["content"]
    assert isinstance(content, list)
    stable = content[0]
    assert stable["cache_control"] == {"type": "ephemeral"}
    assert "GROUND EVERYTHING" in stable["text"]
    # the volatile block follows the cached prefix and carries NO cache_control
    volatile = content[1]
    assert "cache_control" not in volatile


def test_cached_stable_block_excludes_volatile_date_and_selected_blurbs():
    # The cache never hits if volatile content is inside the cached block: the date and the
    # query-selected blurbs must live in the SECOND (uncached) block, not the first.
    agent = _real_agent("anthropic/claude-sonnet-4-6")
    content = agent._system_message("where's the nearest food pantry?")["content"]
    stable_text, volatile_text = content[0]["text"], content[1]["text"]

    assert "Current date & time" not in stable_text
    assert "Current date & time" in volatile_text
    assert "nearest_food_pantry(near=" not in stable_text
    assert "nearest_food_pantry(near=" in volatile_text


def test_system_message_is_plain_string_for_non_anthropic():
    # Every other provider (openai, ollama, ...) keeps the system message as a plain string, with no
    # content blocks and no cache_control, so behavior there is unchanged.
    agent = _real_agent("openai/gpt-4o-mini")
    sysmsg = agent._system_message("where's the nearest food pantry?")

    content = sysmsg["content"]
    assert isinstance(content, str)
    assert "GROUND EVERYTHING" in content              # rules present
    assert "Current date & time" in content            # date present inline (nothing cached)
    assert "nearest_food_pantry(near=" in content      # routed blurb present


def test_build_messages_routes_blurbs_by_query():
    # Progressive disclosure flows through the agent path: a food query loads the food blurb but not
    # the cooling blurb, while the menu + rules are always present.
    agent = _real_agent("openai/gpt-4o-mini")
    system = agent._build_messages("where's the nearest food pantry?", None, None)[0]["content"]

    assert "nearest_food_pantry(near=" in system
    assert "NOT outdoor misting stations" not in system     # cooling blurb not loaded
    assert "Services you can help with (quick menu)" in system
