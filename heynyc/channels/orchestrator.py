"""The universal, channel-agnostic flow: dedup -> rate-limit -> per-user lock ->
bounded concurrency -> run the agent on the user's session -> reply -> record.
Flag keywords short-circuit into the feedback log instead of the agent."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from heynyc.core import outcomes, pii_crypto
from heynyc.core.agent import Agent, _emergency_backstop
from heynyc.core.drafts import DraftStore
from heynyc.core.memory import ContextCapacityError
from heynyc.core.session import Session

from . import analytics
from .base import InboundMessage, KeyedLocks, Replier
from .format import render
from .identity import user_key
from .store import ChannelStore

_FLAG_TOKENS = {"wrong", "report", "incorrect", "bad answer", "👎"}
# Slash forms may carry a free-text reason: `/wrong the hours are outdated`. is_flag still matches
# them, but the pre-consent reason is NOT persisted: the confirmation promises only the last
# exchange is shared, so the reviewer sees exactly that and nothing else.
_FLAG_COMMANDS = ("/wrong", "/report")
_CONFIRM_TOKENS = {"yes", "y"}
# Consent is required, not implied: a flag is only recorded after the resident confirms, and the
# confirmation states exactly what a human will see (the last exchange, nothing else). English-only
# fixed copy, same documented language caveat as the other deterministic command replies.
_FLAG_CONFIRM_MSG = (
    "This will share your last exchange (your message and my reply) with a human reviewer to "
    "improve the service. Nothing else from this conversation is shared. "
    "Reply YES to send, or anything else to cancel."
)
_FLAG_SENT_MSG = "Sent. A human will review that one exchange."
_FLAG_NOTHING_MSG = "Nothing to flag yet, ask me something first."
_HELP_TOKENS = {"hi", "hello", "hey", "help", "menu", "start", "/help", "/menu",
                "what can you do", "what can i ask", "what do you do"}
_RATE_LIMIT_MSG = "You're sending a lot at once, give me a moment and try again shortly. 🙏"
_MEDIA_UNSUPPORTED_MSG = (
    "I received the attachment, but this pilot can't read attachments yet. "
    "Please type the text or question you want help with."
)
_SCREEN_TOOL = "screen_eligibility"
_SCREEN_REMINDER = (
    "The user explicitly requested the official benefits screening action. Build its PII-free "
    "arguments only from the conversation history. Do not invent missing profile details."
)
_NEW_TOKENS = {"new", "/new"}
_PRIVACY_TOKENS = {"privacy", "/privacy"}
# DELETE MY DATA: the fixed self-service-deletion command lane. Like is_flag/is_new, it matches the
# COMMAND by meaning (a curated set of close natural variants) on the whole stripped message, never a
# substring, so a QUESTION about deletion ("how do I delete my data?") is NOT the command — it falls
# through to the agent, which answers from the shipped docs via about_heynyc. Deterministic, no model.
_DELETE_TOKENS = {
    "delete my data", "delete all my data", "delete my info", "delete my information",
    "delete all my info", "delete all my information", "delete my conversation",
    "delete my messages", "delete everything", "erase my data", "erase all my data",
    "erase my info", "erase my information", "erase everything", "forget me", "wipe my data",
}
_DELETE_COMMANDS = ("/delete", "/deletedata", "/forget")
_DELETE_CONFIRM_MSG = (
    "This permanently deletes your data with me: your encrypted conversation transcript, any "
    "queued messages, in-progress application draft, and pending report flags. "
    "What stays: only PII-free aggregate service statistics and an anonymized daily spend record "
    "kept for abuse control, neither of which identifies you. "
    "This can't be undone. Reply YES to delete, or anything else to cancel."
)
_DELETE_DONE_MSG = (
    "Done. I deleted your conversation transcript, queued messages, any application draft, and "
    "pending report flags. All that remains is PII-free aggregate statistics and an anonymized daily spend record "
    "for abuse control, neither of which identifies you. This conversation starts fresh now."
)
# First-contact welcome footer: one line on what HeyNYC is, one naming the controls. Sent once ever.
def _welcome_footer(registry) -> str:
    """First-contact greeting. The capability line derives from the installed manifests at send
    time (the same zero-drift pattern as HELP and the README table), so new modules appear here
    automatically and this copy can never lie about what is installed."""
    categories = sorted({m.category for m in registry.modules if m.category})
    listed = ", ".join(categories[:-1]) + f", and {categories[-1]}" if len(categories) > 1 else (categories[0] if categories else "NYC services")
    return (
        f"First time here? I'm HeyNYC. I help with {listed} across NYC, grounded in real city "
        "data, and I cite my sources.\n"
        "Anytime, text HELP for what I can do, PRIVACY for how your info is handled, REPORT to "
        "flag a bad answer, or DELETE MY DATA to erase everything I keep."
    )
_NEW_MESSAGE = (
    "Started a new conversation. I won't use the earlier chat as context. "
    "This does not delete stored records."
)
_DAILY_CAP_MSG = (
    "You've reached today's usage limit for this number, so I have to pause until midnight. "
    "For city help right now call 311, or 911 in an emergency, and I'll be ready again tomorrow."
)


def _nyc_day() -> str:
    """The current NYC calendar date; the per-resident daily cap resets at NYC midnight."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


@dataclass
class Deps:
    agent: Agent
    store: ChannelStore
    sessions_dir: Path
    salt: str
    telemetry_path: Path
    feedback_path: Path
    locks: KeyedLocks
    semaphore: asyncio.Semaphore
    drafts: Optional[DraftStore] = None   # per-user structured form drafts (None → no persistence)
    # Per-resident, per-NYC-day model-cost ceiling (owner ruling: one resident going ham never
    # dims the service for anyone else). None → off. Emergencies always bypass it.
    user_daily_spend_cap: Optional[float] = None
    # Per-event observer of the SAME stream the drain rides, for a channel that streams a live view
    # (the console REPL). None (Twilio/Meta) is byte-identical to today: no view, pure drain.
    event_sink: Optional[Callable[[object], None]] = None


def is_flag(text: str) -> bool:
    """The user is flagging the last answer as wrong: a bare token (`wrong`, `👎`) or a slash
    command (`/wrong`, `/report`) that may carry an optional free-text reason. Routed to the
    feedback log, never the agent. A sentence that merely contains 'wrong' is NOT a flag."""
    t = text.strip().lower()
    return t in _FLAG_TOKENS or any(t == cmd or t.startswith(cmd + " ") for cmd in _FLAG_COMMANDS)


def flag_note(text: str) -> str:
    """The free-text reason after a `/wrong` / `/report` command, else '' (bare tokens carry none).
    Not persisted under the consent gate (the confirmation shares only the last exchange), kept as a
    pure parser for the CLI / future reason-capture behind explicit consent."""
    t = text.strip()
    low = t.lower()
    for cmd in _FLAG_COMMANDS:
        if low.startswith(cmd + " "):
            return t[len(cmd):].strip()
    return ""


def is_help(text: str) -> bool:
    """A greeting / 'what can you do', answered with the grounded capability menu, not the agent."""
    return text.strip().lower().rstrip("!?. ") in _HELP_TOKENS


def is_screen(text: str) -> bool:
    """The exact explicit action command, never a guess from ordinary conversation."""
    return text.strip().lower() in {"/screen", "/screen all"}


def is_confirm(text: str) -> bool:
    """An affirmative reply to a pending confirmation (the flag consent gate). Only matched while a
    confirmation is actually pending, so a stray 'yes' in normal chat is a plain turn."""
    return text.strip().lower().rstrip("!?. ") in _CONFIRM_TOKENS


def is_new(text: str) -> bool:
    return text.strip().lower() in _NEW_TOKENS


def is_privacy(text: str) -> bool:
    return text.strip().lower() in _PRIVACY_TOKENS


def is_delete(text: str) -> bool:
    """The fixed DELETE MY DATA command: an exact command phrase (a curated close-variant set) or a
    slash form (`/delete`, optionally with trailing text). Matched only on the whole stripped
    message, so a QUESTION about deletion routes to the agent instead (see is_flag for the pattern)."""
    t = text.strip().lower().rstrip("!?. ")
    return t in _DELETE_TOKENS or any(t == cmd or t.startswith(cmd + " ") for cmd in _DELETE_COMMANDS)


def _privacy_message(channel: str) -> str:
    days = pii_crypto.retention_days()
    retention = str(int(days)) if days.is_integer() else str(days)
    delivery = "Twilio" if "twilio" in channel else "Meta"
    return (
        "HeyNYC keeps an encrypted conversation transcript and any unfinished application draft "
        f"for up to {retention} days by default so a conversation can continue after a restart. "
        f"Messages needed for a reply go to the configured AI model provider and {delivery} "
        "delivers the reply. HeyNYC uses a pseudonymous sender key, not your raw phone number, in "
        "its own session and operational logs. Do not send an SSN or other sensitive ID in chat. "
        "Send NEW to start without earlier model context. If I get something wrong, reply REPORT "
        "and, after you confirm, that one exchange is shared with a human reviewer, nothing else. "
        "To erase your data, send DELETE MY DATA and confirm: that deletes your transcript, any "
        "queued messages, draft, and pending flags, keeping only PII-free aggregate stats and an anonymized "
        "daily spend record for abuse control."
    )


def _reminders() -> list[str]:
    return [f"Today's date is {datetime.now():%A, %B %-d, %Y}. The user is in New York City."]


def _last(turns: list[dict], role: str) -> str:
    for turn in reversed(turns):
        if turn.get("role") == role:
            return turn.get("content") or ""
    return ""


def _artifacts_in(art_dir: Path) -> list[str]:
    """Files a tool wrote into THIS request's artifacts dir. The orchestrator owns this directory,
    so untrusted tool-result content can never influence which file is read/sent, we list the dir,
    we never parse a path out of model/tool text (that would be an arbitrary-file-read sink)."""
    return sorted(str(p) for p in art_dir.glob("*") if p.is_file())


async def handle(
    msg: InboundMessage, replier: Replier, deps: Deps, *, deduplicate: bool = True,
) -> None:
    key = user_key(msg.channel, msg.sender, deps.salt)
    if deduplicate and deps.store.seen(msg.message_id, key):  # records before any work
        return
    if not deps.store.allow(key):
        await replier.send_text(_RATE_LIMIT_MSG)
        return

    async with deps.locks.get(key):           # serialize one user's messages
        async with deps.semaphore:            # bound global concurrency / LLM spend
            session = Session.load(deps.agent, key, deps.sessions_dir / f"{key}.jsonl")
            # Consent gate for REPORT and DELETE MY DATA: a prior command staged a pointer / a
            # deletion awaiting YES. Both pending states are consumed on EVERY message, so any
            # non-YES reply expires them and is then handled as an ordinary turn. Only one is ever
            # outstanding (each fresh command cancels the other), and DELETE takes priority.
            staged_flag = deps.store.pop_pending_flag(key)
            staged_delete = deps.store.pop_pending_delete(key)
            if is_confirm(msg.text):
                if staged_delete is not None:
                    _execute_delete(key, deps)
                    await replier.send_text(_DELETE_DONE_MSG)
                    return
                if staged_flag is not None:
                    _confirm_flag(msg, key, session, staged_flag, deps)
                    await replier.send_text(_FLAG_SENT_MSG)
                    return
            if msg.media:
                emergency_response = _emergency_backstop(msg.text)
                if emergency_response:
                    await replier.send_text(emergency_response)
                    return
                await replier.send_text(_MEDIA_UNSUPPORTED_MSG)
                return
            if is_flag(msg.text):
                await _handle_flag(msg, key, session, replier, deps)
                return
            if is_new(msg.text):
                await replier.send_text(_NEW_MESSAGE)
                session.reset()
                return
            if is_privacy(msg.text):
                await replier.send_text(_privacy_message(msg.channel))
                return
            if is_delete(msg.text):   # DELETE MY DATA → stage the confirmation, delete only on YES
                deps.store.set_pending_delete(key)
                await replier.send_text(_DELETE_CONFIRM_MSG)
                return
            if is_help(msg.text):   # greeting / "what can you do" → the grounded capability menu
                await replier.send_text(deps.agent.registry.welcome_text())
                return
            # Per-resident daily cost cap, after the free commands (they stay available while
            # capped) and only when the deterministic emergency backstop would not fire: a
            # crisis message always reaches the zero-cost emergency path.
            if (
                deps.user_daily_spend_cap is not None
                and _emergency_backstop(msg.text) is None
                and deps.store.daily_spend(key, _nyc_day()) >= deps.user_daily_spend_cap
            ):
                await replier.send_text(_DAILY_CAP_MSG)
                return
            await replier.indicate_typing()
            art_dir = Path(tempfile.mkdtemp(prefix="heynyc-art-"))   # per-request, orchestrator-owned
            try:
                user_drafts = deps.drafts.for_user(key) if deps.drafts else None
                screen_requested = is_screen(msg.text)
                reminders = _reminders() + ([_SCREEN_REMINDER] if screen_requested else [])
                try:
                    pending = await session.prepare(
                        msg.text, reminders=reminders, output_dir=art_dir, drafts=user_drafts,
                        forced_tool=_SCREEN_TOOL if screen_requested else None,
                        forced_tool_args={
                            "show_all": msg.text.strip().lower() == "/screen all",
                        } if screen_requested else None,
                        excluded_tools=None if screen_requested else {_SCREEN_TOOL},
                        event_sink=deps.event_sink,
                    )
                except ContextCapacityError:
                    await replier.send_text(
                        "I can't safely fit enough of this conversation into the AI model right "
                        "now. Please try again shortly or send NEW to start a fresh conversation."
                    )
                    return
                result = pending.result
                # First-contact welcome LEADS on every channel (owner, 2026-07-21: greet first,
                # then answer, the way a person would; trailing it after sources read as an
                # afterthought). Once ever, marked only on the answer path so a first message
                # that is a command isn't spent on it. The console banner merely lists commands;
                # this explains them, so the console gets it too.
                if deps.store.first_contact(key):
                    await replier.send_text(_welcome_footer(deps.agent.registry) + "\n\nNow, about your message:")
                for chunk in render(result, msg.channel):
                    await replier.send_text(chunk)
                artifacts = _artifacts_in(art_dir)    # only files the tool wrote into OUR dir
                for path in artifacts:
                    await replier.send_document(path, caption="Your draft SNAP application (LDSS-4826)")
                finalize = getattr(replier, "finalize", None)
                if finalize is not None:
                    await finalize()
                session.commit(pending)
                turn_cost = result.usage.get("cost_usd")
                deps.store.add_spend(key, _nyc_day(), float(turn_cost) if turn_cost else 0.0)
                analytics.record_interaction(
                    telemetry_path=deps.telemetry_path, model=deps.agent.model,
                    user_key=key, channel=msg.channel, result=result,
                )
                # --- OUTCOMES FUNNEL HOOK (OTI Gap 5) ---------------------------------
                # telemetry already carries which tools fired (screened / apply-started).
                # These two milestones need the tool RESULT, which telemetry lacks, so
                # record them here, PII-free (user_key + two booleans), into the sidecar.
                outcomes.record_milestone(
                    outcomes.default_path(deps.telemetry_path.parent), user_key=key,
                    **outcomes.milestones_from_result(result, produced_artifact=bool(artifacts)),
                )
            finally:
                shutil.rmtree(art_dir, ignore_errors=True)


def _execute_delete(key: str, deps: Deps) -> None:
    """Irreversibly delete this resident's own data on a confirmed DELETE MY DATA: their encrypted
    session transcript (the JSONL file), any in-progress application draft (the draft file), and any
    report-flag rows (pending + confirmed). PII-free aggregate statistics and the anonymized daily
    spend record survive for abuse control, exactly as the confirmation copy promised. Deleting the
    session file means the next inbound message loads an empty history, so the conversation is fresh."""
    (deps.sessions_dir / f"{key}.jsonl").unlink(missing_ok=True)
    if deps.drafts is not None:
        deps.drafts.for_user(key).delete()
    deps.store.delete_user(key)


def _flag_token(text: str) -> str:
    """The bounded command/token the resident used (`report`, `👎`, `/wrong`, ...), never the
    free-text reason. Safe to store: it is a control word, not message content."""
    low = text.strip().lower()
    return next((cmd for cmd in _FLAG_COMMANDS if low.startswith(cmd)), text.strip())


async def _handle_flag(msg, key, session, replier, deps) -> None:
    """Step one of the consent-gated flag: stage a POINTER to the last exchange and ask the
    resident to confirm. Nothing is recorded until they reply YES (see the gate in handle)."""
    turn_index = max(
        (i for i, turn in enumerate(session.turns) if turn.get("role") == "assistant"),
        default=-1,
    )
    if turn_index < 0:
        await replier.send_text(_FLAG_NOTHING_MSG)
        return
    deps.store.set_pending_flag(key, turn_index, _flag_token(msg.text))
    await replier.send_text(_FLAG_CONFIRM_MSG)


def _confirm_flag(msg, key, session, staged, deps) -> None:
    """Record the consented flag: a content-free POINTER for triage (the encrypted session JSONL
    holds the turns), plus the pre-existing redacted aggregate record. Scope is exactly the one
    exchange the resident agreed to share, so the pre-consent free-text reason is dropped."""
    turn_index = staged["turn_index"]
    turns = session.turns
    agent_text = turns[turn_index].get("content", "") if 0 <= turn_index < len(turns) else ""
    user_query = (
        turns[turn_index - 1].get("content", "")
        if 0 < turn_index < len(turns) and turns[turn_index - 1].get("role") == "user"
        else ""
    )
    deps.store.add_flag(user_key=key, turn_index=turn_index, flag=staged["flag"])
    analytics.record_feedback(
        deps.feedback_path, user_key=key, channel=msg.channel, message_id=msg.message_id,
        flag=staged["flag"], note="", user_query=user_query, agent_text=agent_text,
    )
