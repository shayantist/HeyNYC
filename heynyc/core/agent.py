"""HeyNYC agent — a grounded, streaming tool-calling harness.

The core is the standard agent loop (LLM + tools until no tool calls), but built
as an event stream so a UI can show work in progress, with model-call retries,
clean terminal events, reactive system-reminders, and minimal approval gating for
side-effecting tools. `run()` is a convenience that drains the stream into a result.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional

from . import config, events
from .citations import CitationRegistry
from .grounding import GroundingResult, check_grounding
from .prompts import build_system_prompt_tiers
from .registry import Registry
from .tools import Tool, ToolContext, build_toolbox

logger = logging.getLogger("heynyc.agent")

# Engine default. The application injects its configured model via `Agent(model=...)`;
# the core does NOT read a domain config module, so it stays reusable across projects.
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

# Safe fallback for a terminal turn (no tool calls) that comes back empty/whitespace. Some inputs —
# notably an encoded-instruction injection the model refuses by going silent — yield a blank final
# turn; the user must NEVER see an empty response, so we substitute an explicit safe refusal.
EMPTY_ANSWER_FALLBACK = (
    "I can't help with that request. If there's something about NYC services, benefits, or events "
    "I can help you find, tell me in your own words and I'll do my best — or you can call 311."
)

# --- Deterministic grounding guard (post-generation safety hook) ----------------------------------
# The single most important safety mechanism for running HeyNYC on a cheaper model: after the agent
# produces its FINAL answer, we deterministically re-check that every {cite:Sn}'d structured fact
# actually appears in the source it's attributed to (core.grounding.check_grounding — the SAME logic
# the eval gate uses). A HARD mismatch (a verbatim phone / dollar amount / address absent from an
# all-complete-capture source) is a fabrication: we feed the model a SPECIFIC correction and let it
# regenerate (Tier 3), capped; if it still can't ground it, we strip the offending claim or abstain and
# route to 311 (Tier 4). SOFT mismatches (name drift, or anything cited to a truncated snippet) never
# fire — preserving the check's zero-false-fail calibration so a CORRECT answer is never over-blocked.
GUARD_MAX_RETRIES = 2

# Tier 4 last resort: the model could not ground a load-bearing fact after the retry cap. Rather than
# ship an unverified number/address, hold off and route to a human / the official source.
GROUNDING_ABSTAIN_FALLBACK = (
    "I want to get this right, and I couldn't confirm that detail against an official source — so I'd "
    "rather not guess and risk sending you the wrong number or address. For the accurate, current "
    "info, call 311 or check the official NYC page, and they can take it from there."
)

_CITE_STRIP_RE = re.compile(r"\{cite:S\d+\}")


def _grounding_feedback(result: GroundingResult) -> str:
    """The SPECIFIC correction fed back to the model (Tier 3). Names each ungrounded fact so the model
    fixes THAT fact rather than blindly rewriting or over-abstaining."""
    problems = "; ".join(m.message for m in result.hard_failures)
    return (
        "<system-reminder>\n"
        "A grounding check ran on your last answer and found a cited fact your source does not "
        f"support: {problems}. Each is a structured fact (a phone number, dollar amount, address, or "
        "the like) you attributed to a {cite:Sn} source whose content does not contain it. Fix it: "
        "correct the fact to exactly match the cited source, cite a source that actually contains it, "
        "remove the claim, or — if you can't ground it — abstain on that specific fact and point the "
        "user to 311 or the official page. Do not repeat the unsupported fact.\n"
        "</system-reminder>"
    )


def _strip_ungrounded_claims(text: str, result: GroundingResult) -> str:
    """Tier 4: remove the sentence(s) carrying an ungrounded structured fact; if that guts the answer
    (nothing substantive left), return the abstention that routes to 311. Otherwise return the answer
    with the offending claim(s) — and their now-orphaned citations — excised."""
    stripped = text
    for claim in {m.claim for m in result.hard_failures}:
        if claim and claim in stripped:
            stripped = stripped.replace(claim, " ")
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r" *\n *", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    meaningful = _CITE_STRIP_RE.sub("", stripped).strip()
    if len(meaningful) < 40:  # the ungrounded fact was load-bearing → the whole answer must abstain
        return GROUNDING_ABSTAIN_FALLBACK
    return stripped

# Non-streaming model fn: (messages, tool_schemas) -> assistant message dict.
CompletionFn = Callable[[list[dict], list[dict]], Awaitable[dict]]
# Streaming model fn: yields {"type":"text","text":...} deltas then a terminal
# {"type":"message","message": {role, content, tool_calls}}.
StreamFn = Callable[[list[dict], list[dict]], AsyncIterator[dict]]
# Approval callback for side-effecting tools: (name, args) -> approved?
Approver = Callable[[str, dict], Awaitable[bool]]


@dataclass
class AgentResult:
    text: str
    citations: dict[str, dict]
    tool_calls_made: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_max_iters: bool = False
    status: str = "success"
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens, latency_ms} per turn


class Agent:
    def __init__(
        self,
        registry: Registry,
        tools: Optional[dict[str, Tool]] = None,
        model: Optional[str] = None,
        complete_fn: Optional[CompletionFn] = None,
        stream_fn: Optional[StreamFn] = None,
        approver: Optional[Approver] = None,
        index=None,
        guard_grounding: bool = True,
        guard_max_retries: int = GUARD_MAX_RETRIES,
    ):
        self.registry = registry
        self._embedder = getattr(index, "embedder", None)  # shared with retrieval-using module tools
        self.tools = tools if tools is not None else build_toolbox(registry, index=index)
        self.model = model or DEFAULT_MODEL
        self._approver = approver
        # Deterministic post-generation grounding guard. On by default — it's the safety floor that lets
        # HeyNYC run on a cheaper model; disable only to observe raw model output (see tests).
        self.guard_grounding = guard_grounding
        self.guard_max_retries = guard_max_retries
        if stream_fn is not None:
            self._stream_fn = stream_fn
        elif complete_fn is not None:
            self._stream_fn = _wrap_complete(complete_fn)
        else:
            self._stream_fn = self._litellm_stream

    def _tool_schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def _grounding_verdict(self, text: str, citations_map: dict, query: str, retries: int):
        """Run the deterministic grounding guard on a terminal answer. Returns
        (action, out_text, result):
          "pass"    → ship out_text unchanged (nothing to verify, or only SOFT mismatches)
          "retry"   → feed _grounding_feedback(result) back and let the model regenerate (Tier 3)
          "abstain" → ship out_text (offending claim stripped, or the abstention fallback) (Tier 4)
        The guard acts ONLY on a HARD (blocking) mismatch — a verbatim structured fact absent from an
        all-complete-capture source. Soft mismatches pass through, so a correct answer is never
        over-blocked."""
        if not self.guard_grounding:
            return "pass", text, None
        result = check_grounding(text, citations_map, query)
        if result is None or not result.blocking:
            return "pass", text, result
        if retries < self.guard_max_retries:
            return "retry", text, result
        return "abstain", _strip_ungrounded_claims(text, result), result

    def _system_message(self, query: str) -> dict:
        """The system message for THIS turn. The stable tier (safety rules + capability menu) is a
        cacheable prefix; the volatile tier (query-selected blurbs + the date) follows it.

        For an Anthropic model we emit `content` as two text blocks and mark the STABLE block with
        cache_control so repeat calls read that prefix from cache (~90% cheaper). Every other provider
        (openai, ollama, ...) gets a plain-string `content`, unchanged. This is purely a transport /
        caching choice: it never changes which safety rules or tools ship. Routing selects the
        detailed blurbs identically for both shapes, and a routing miss can only drop a blurb."""
        stable, volatile = build_system_prompt_tiers(self.registry, query=query)
        if _is_anthropic(self.model):
            return {"role": "system", "content": [
                {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": volatile},
            ]}
        return {"role": "system", "content": stable + volatile}

    def _build_messages(self, user_message: str, history, reminders) -> list[dict]:
        messages: list[dict] = [self._system_message(user_message)]
        messages.extend(history or [])
        for reminder in reminders or []:
            messages.append({"role": "user", "content": f"<system-reminder>\n{reminder}\n</system-reminder>"})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def stream(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        max_iters: int = 8,
        reminders: Optional[list[str]] = None,
        output_dir=None,
        drafts=None,
    ) -> AsyncIterator[events.Event]:
        """Run one turn, yielding events (text deltas, tool lifecycle, terminal done)."""
        messages = self._build_messages(user_message, history, reminders)
        citations = CitationRegistry()
        ctx = ToolContext(citations=citations, registry=self.registry, embedder=self._embedder,
                          output_dir=output_dir, drafts=drafts)
        tools_made: list[str] = []
        guard_retries = 0  # how many times the grounding guard has bounced a terminal answer back
        turn_started = time.perf_counter()
        turn_usage = {"input_tokens": 0, "output_tokens": 0}

        def _usage() -> dict:
            return {**turn_usage, "latency_ms": (time.perf_counter() - turn_started) * 1000.0}

        for reminder in reminders or []:
            yield events.Reminder(summary=reminder)

        for i in range(max_iters):
            message_id = f"m{i}"
            yield events.MessageStart(message_id=message_id)
            parts: list[str] = []
            assistant: Optional[dict] = None
            try:
                async for chunk in self._stream_fn(messages, self._tool_schemas()):
                    if chunk["type"] == "text":
                        parts.append(chunk["text"])
                        yield events.TextDelta(message_id=message_id, text=chunk["text"])
                    elif chunk["type"] == "usage":
                        turn_usage["input_tokens"] += int(chunk.get("input_tokens", 0) or 0)
                        turn_usage["output_tokens"] += int(chunk.get("output_tokens", 0) or 0)
                    elif chunk["type"] == "message":
                        assistant = chunk["message"]
            except Exception as exc:  # model call failed after retries
                logger.exception("model stream failed")
                yield events.ErrorEvent(scope="model", message=str(exc), retryable=True)
                result = AgentResult(
                    text="", citations=citations.mapping(), tool_calls_made=tools_made,
                    iterations=i, status="error", messages=messages, usage=_usage(),
                )
                yield events.Done(status="error", num_turns=i, citations=result.citations, result=result)
                return

            if assistant is None:
                assistant = {"role": "assistant", "content": "".join(parts) or None, "tool_calls": None}
            tool_calls = assistant.get("tool_calls") or []
            text = assistant.get("content") or "".join(parts)
            # EMPTY-ANSWER GUARD: a terminal turn (no tool calls) must never reach the user blank.
            # Substitute an explicit safe refusal and stream it, so both the streaming UI and the
            # drained result get non-empty, safe text.
            if not tool_calls and not (text or "").strip():
                text = EMPTY_ANSWER_FALLBACK
                assistant["content"] = text
                yield events.TextDelta(message_id=message_id, text=text)
            messages.append(assistant)

            if not tool_calls:
                # DETERMINISTIC GROUNDING GUARD (the post-generation safety hook). Before the final
                # answer reaches the user, verify every {cite:Sn}'d structured fact is in its source.
                action, text, gr = self._grounding_verdict(
                    text, citations.mapping(), user_message, guard_retries)
                if action == "retry":
                    # Tier 3: a cited structured fact isn't grounded. Close out the rejected attempt
                    # for streaming clients, tell the model EXACTLY what's wrong, and regenerate.
                    guard_retries += 1
                    yield events.MessageCompleted(message_id=message_id,
                                                  text=assistant.get("content") or "",
                                                  citations=citations.mapping())
                    yield events.Reminder(summary=("grounding guard: unsupported cited fact, retrying "
                                                   f"({guard_retries}/{self.guard_max_retries})"))
                    messages.append({"role": "user", "content": _grounding_feedback(gr)})
                    continue
                # "pass" (grounded, or only soft mismatches) or "abstain" (Tier 4 rewrote `text`).
                assistant["content"] = text
                yield events.MessageCompleted(message_id=message_id, text=text, citations=citations.mapping())
                result = AgentResult(
                    text=text, citations=citations.mapping(), tool_calls_made=tools_made,
                    iterations=i + 1, status="success", messages=messages, usage=_usage(),
                )
                yield events.Done(status="success", num_turns=i + 1, citations=result.citations, result=result)
                return

            yield events.MessageCompleted(message_id=message_id, text=text, citations=citations.mapping())
            for call in tool_calls:
                name = call["function"]["name"]
                call_id = call.get("id") or name
                tools_made.append(name)
                tool = self.tools.get(name)
                yield events.ToolStart(tool_call_id=call_id, name=name, label=name)

                async for ev, tool_result in self._invoke(name, call["function"]["arguments"], tool, ctx):
                    if tool_result is None:
                        yield ev  # an approval-required event
                        continue
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                    status = "error" if tool_result.startswith(("ERROR", "Action not approved")) else "ok"
                    yield events.ToolCompleted(
                        tool_call_id=call_id, name=name, status=status, result_summary=tool_result[:200]
                    )

        result = AgentResult(
            text=(messages[-1].get("content") or "").strip() or EMPTY_ANSWER_FALLBACK,
            citations=citations.mapping(),
            tool_calls_made=tools_made, iterations=max_iters, hit_max_iters=True,
            status="max_turns", messages=messages, usage=_usage(),
        )
        yield events.Done(status="max_turns", num_turns=max_iters, citations=result.citations, result=result)

    async def run(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        max_iters: int = 8,
        reminders: Optional[list[str]] = None,
        output_dir=None,
        drafts=None,
    ) -> AgentResult:
        """Drain the event stream into a single AgentResult."""
        result: Optional[AgentResult] = None
        async for event in self.stream(user_message, history=history, max_iters=max_iters,
                                       reminders=reminders, output_dir=output_dir, drafts=drafts):
            if isinstance(event, events.Done):
                result = event.result
        assert result is not None  # stream always ends with Done
        return result

    async def _invoke(self, name: str, raw_args, tool: Optional[Tool], ctx: ToolContext):
        """Yield (event, tool_result). tool_result is None for non-terminal events
        (e.g. approval prompts); a string when the call resolved."""
        if tool is None:
            yield None, f"ERROR: unknown tool '{name}'."
            return
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            yield None, f"ERROR: could not parse arguments for '{name}': {exc}"
            return

        if tool.requires_approval:
            yield events.ToolApprovalRequired(tool_call_id=name, name=name, args=args), None
            approved = await self._approver(name, args) if self._approver else False
            if not approved:
                yield None, "Action not approved by the user; not executed."
                return

        try:
            yield None, await tool.handler(args, ctx)
        except Exception as exc:  # surface tool errors to the model, don't crash
            logger.exception("tool %s failed", name)
            yield None, f"ERROR: tool '{name}' failed: {exc}"

    def conversation(self) -> "Conversation":
        return Conversation(self)

    async def _litellm_stream(self, messages: list[dict], tool_schemas: list[dict]) -> AsyncIterator[dict]:
        import litellm

        kwargs = _completion_kwargs(self.model, messages, tool_schemas)

        async def _open():
            return await litellm.acompletion(**kwargs)

        stream = await _with_retry(_open)
        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: Optional[dict] = None
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = {"input_tokens": chunk.usage.prompt_tokens,
                         "output_tokens": chunk.usage.completion_tokens}
            if not chunk.choices:  # include_usage emits a final choices-less chunk
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield {"type": "text", "text": delta.content}
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if usage is not None:
            yield {"type": "usage", **usage}
        tool_calls = None
        if calls:
            tool_calls = [
                {"id": s["id"] or f"call_{i}", "type": "function",
                 "function": {"name": s["name"], "arguments": s["args"]}}
                for i, s in sorted(calls.items())
            ]
        yield {"type": "message", "message": {"role": "assistant", "content": "".join(content_parts) or None, "tool_calls": tool_calls}}


def _is_anthropic(model: str) -> bool:
    """Anthropic-family model? Only these accept prompt-cache content blocks (cache_control on a
    system block); every other provider must get a plain-string system message. Detected from the
    model string so it also covers Bedrock / Vertex Claude ids (e.g. 'bedrock/anthropic.claude-...')."""
    m = model.lower()
    return m.startswith("anthropic/") or "claude" in m


def _completion_kwargs(model: str, messages: list[dict], tool_schemas: list[dict]) -> dict:
    """Build the kwargs for litellm.acompletion.

    GPT-5 models reject temperature != 1 (litellm raises UnsupportedParamsError), so we omit
    temperature for them and let it default; every other model pins temperature=0 for
    deterministic, grounded output. Tool schemas are attached only when present.
    """
    kwargs: dict = {
        "model": model, "messages": messages,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if "gpt-5" not in model:
        kwargs["temperature"] = 0.0
    if tool_schemas:
        kwargs["tools"] = tool_schemas
    if "ollama" in model:
        kwargs["num_ctx"] = config.OLLAMA_NUM_CTX
    return kwargs


def _wrap_complete(fn: CompletionFn) -> StreamFn:
    async def _stream(messages: list[dict], tool_schemas: list[dict]) -> AsyncIterator[dict]:
        message = await fn(messages, tool_schemas)
        if message.get("content"):
            yield {"type": "text", "text": message["content"]}
        yield {"type": "message", "message": message}

    return _stream


async def _with_retry(factory, attempts: int = 3, base_delay: float = 0.5):
    delay = base_delay
    for attempt in range(attempts):
        try:
            return await factory()
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


class Conversation:
    """Stateful multi-turn wrapper. Keeps user/assistant turns as history so the
    agent has context across messages. Tool-call noise stays within a single turn."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.turns: list[dict] = []

    async def send(self, user_message: str, max_iters: int = 8, reminders=None,
                   output_dir=None, drafts=None) -> AgentResult:
        result = await self.agent.run(user_message, history=self.turns, max_iters=max_iters,
                                      reminders=reminders, output_dir=output_dir, drafts=drafts)
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": result.text})
        return result

    async def stream(self, user_message: str, max_iters: int = 8, reminders=None,
                     output_dir=None, drafts=None):
        """Stream a turn's events, then commit the turn to history."""
        final_text = ""
        async for event in self.agent.stream(user_message, history=self.turns, max_iters=max_iters,
                                             reminders=reminders, output_dir=output_dir, drafts=drafts):
            if isinstance(event, events.Done) and event.result is not None:
                final_text = event.result.text
            yield event
        self.turns.append({"role": "user", "content": user_message})
        self.turns.append({"role": "assistant", "content": final_text})
