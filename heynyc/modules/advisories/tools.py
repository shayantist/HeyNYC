"""advisories module tool: `check_notify_nyc`, grounded in the live Notify NYC feed.

Sourcing is TIERED and fail-safe (see heynyc/core/tools/notify_nyc.py):
  1. The Everbridge RSS feed of CAP alerts is the STRUCTURED source (severity + "in effect until" +
     areaDesc), each reported with a DATA citation to its resolvable CAP XML url. Used WHEN it works.
  2. The city's own live "recent messages" endpoint (what the Notify NYC portal itself renders) is
     the REAL-TIME fallback used when the CAP feed is empty or unreachable, which has been the actual
     failure mode: the CAP feed was publishing empty even mid-emergency. It is thinner (no
     machine-readable severity/expiry), so we surface each notification's ISSUE time and its body,
     cited to the resolvable endpoint.
  3. FAIL-SAFE: if NEITHER source can be confirmed, we NEVER say "no advisories". We say we could not
     confirm and route to the official live source (nyc.gov/notifynyc) + 311, and 911 for a
     life-threatening emergency. A confident false all-clear off an empty/down feed is the bug this
     module exists to prevent.

Honest limitations: the CAP feed's geography is often citywide even for a local event, so we do NOT
filter out citywide alerts on a `near` hint; either source can lag the SMS/email alerts by minutes;
and the fallback has no structured expiry, so we show the issue time and never invent one. We never
invent an advisory, a severity, or an expiry.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from time import monotonic
from zoneinfo import ZoneInfo

from heynyc.core.citations import CitationRegistry, data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.notify_nyc import (
    RECENT_MESSAGES_URL,
    RSS_URL,
    Advisory,
    RecentFeed,
    RecentNote,
    active_advisories,
    fetch_recent_advisories,
)

NOTIFY_NYC_URL = "https://www.nyc.gov/notifynyc"
OFFICIAL = f"Notify NYC ({NOTIFY_NYC_URL}) or call 311"
NYC_TZ = ZoneInfo("America/New_York")
PLAN_RELEVANCE = (
    "When the resident asks which notices could affect a current or future plan, compare each "
    "notice's stated area and time with the requested place and date. Do not enumerate notices "
    "that clearly do not overlap. If none overlap, say so plainly and name the date and place "
    "you checked."
)
_AWARENESS_TTL_S = 60.0
_AWARENESS_MAX_CHARS = 16_000
_awareness_cache: tuple[float, dict[str, RecentNote]] | None = None
_awareness_lock = asyncio.Lock()

# CONFIRMED all-clear: the feed was reached and read, and nothing is currently in effect. Only this
# state may tell the user there are no active advisories.
NO_ACTIVE = (
    "The Notify NYC feed was reached and read, and it shows no advisories active right now. Tell the "
    "user there are no active Notify NYC advisories at the moment (this is the public Notify NYC feed, "
    f"not the whole picture) and point them to {OFFICIAL}. Do NOT invent an advisory. Offer to check "
    "again."
)

# DEGRADED / FAIL-SAFE: the feed was unreachable, errored, empty, or unreadable. We could NOT confirm
# the current advisories, so we must NEVER announce that there are none. This is the safety-critical
# branch: a false "no advisories" during a live weather emergency could get someone hurt.
COULD_NOT_CONFIRM = (
    "We could not reach or read the Notify NYC feed just now, so we could NOT confirm the current "
    "advisories. Do NOT tell the user the city is clear or that nothing is active, and do NOT state "
    "an all-clear. Tell them plainly that the current advisory status is unknown, then send them to "
    f"the official live source, Notify NYC at {NOTIFY_NYC_URL}, and to 311 for current advisories. "
    "For a life-threatening emergency, tell "
    "them to call 911 right away. Offer to check again."
)


def _cap_notice_id(advisory: Advisory) -> str:
    return f"cap:{advisory.guid}"


def _selection_receipt_citation(
    ctx: ToolContext,
    query: str,
    selected_notice_ids: list[str],
    *,
    cap_confirmed: bool,
    recent_confirmed: bool,
    cap_records: int,
    recent_records: int,
    observed_at: datetime,
) -> str:
    snapshot = {
        "observed_at": observed_at.isoformat(),
        "query": query,
        "selected_notice_ids": selected_notice_ids,
        "sources": [
            {"url": RSS_URL, "confirmed": cap_confirmed},
            {"url": RECENT_MESSAGES_URL, "confirmed": recent_confirmed},
        ],
        "counts": {
            "cap_records": cap_records,
            "recent_records": recent_records,
            "selected_records": len(selected_notice_ids),
        },
    }
    provenance = data_provenance(
        snapshot,
        record_id=f"notify-nyc-selection-{observed_at.isoformat()}",
        field_pointer="/",
        derivation=snapshot,
    )
    return ctx.citations.register(
        NOTIFY_NYC_URL,
        snippet=(
            f"No current Notify NYC notice was selected for {query or 'the resident request'}."
        ),
        title="Notify NYC current-notice selection",
        kind="DATA",
        valid_as_of=observed_at.isoformat(),
        provenance=provenance,
    )


def _advisory_snapshot(advisory: Advisory) -> dict:
    """The exact fields cited, hashed into the DATA citation for reproducibility/integrity."""
    return {
        "identifier": advisory.guid,
        "sender": advisory.sender,
        "status": advisory.status,
        "msgType": advisory.message_type,
        "scope": advisory.scope,
        "references": list(advisory.references),
        "language": advisory.language,
        "headline": advisory.headline,
        "event": advisory.event,
        "category": advisory.category,
        "severity": advisory.severity,
        "urgency": advisory.urgency,
        "certainty": advisory.certainty,
        "effective": advisory.effective,
        "onset": advisory.onset,
        "description": advisory.description,
        "instruction": advisory.instruction,
        "responseType": list(advisory.response_types),
        "sent": advisory.sent,
        "expires": advisory.expires,
        "areaDesc": advisory.area_desc,
    }


def _advisory_citation(ctx: ToolContext, advisory: Advisory) -> str:
    """Register a DATA citation grounded in the advisory's resolvable CAP XML (re-fetchable + hashed).
    `valid_as_of` is the advisory's own `sent` time, temporal provenance, never fetch time."""
    normalized = _advisory_snapshot(advisory)
    provenance = data_provenance(
        advisory.provider_record or normalized,
        record_id=advisory.guid,
        field_pointer="/",
        derivation=normalized,
    )
    return ctx.citations.register(
        advisory.source_url,
        snippet=" ".join(filter(None, (
            f"{advisory.headline or advisory.event}, in effect until {advisory.expires}.",
            f"Area: {advisory.area_desc or 'unknown'}.",
            advisory.description,
            advisory.instruction,
        ))),
        title=advisory.headline or advisory.event or "Notify NYC advisory",
        kind="DATA",
        valid_as_of=advisory.sent,
        provenance=provenance,
    )


def _advisory_block(advisory: Advisory, cite: str) -> str:
    headline = advisory.headline or advisory.event or "NYC advisory"
    severity = advisory.severity or "Unknown"
    parts = [
        f"- {headline} [{severity}], {advisory.event or 'advisory'}, "
        f"in effect until {advisory.expires} {{cite:{cite}}}"
    ]
    parts.append(f"  Area (per feed): {advisory.area_desc or 'NYC'}")
    if advisory.category:
        parts.append(f"  Category: {advisory.category}")
    if advisory.certainty:
        parts.append(f"  Certainty: {advisory.certainty}")
    if advisory.effective:
        parts.append(f"  Effective: {advisory.effective}")
    if advisory.onset:
        parts.append(f"  Onset: {advisory.onset}")
    if advisory.response_types:
        parts.append(f"  Response types: {', '.join(advisory.response_types)}")
    if advisory.description:
        parts.append(f"  {advisory.description}")
    if advisory.instruction:
        parts.append(f"  Instruction: {advisory.instruction}")
    parts.append(f"  As of: {advisory.sent or 'unknown'}")
    return "\n".join(parts)


def _register_recent_citation(
    citations: CitationRegistry,
    note: RecentNote,
) -> str:
    """DATA citation for a live "recent messages" note, grounded in the resolvable Notify NYC endpoint.
    `valid_as_of` is the notification's own ISSUE time (temporal provenance), never fetch time."""
    normalized = {"title": note.title, "body": note.body, "pubDate": note.issued_raw}
    snapshot = note.provider_record or normalized
    provenance = data_provenance(
        snapshot,
        record_id=note.guid,
        field_pointer="/",
        derivation=normalized,
    )
    return citations.register(
        note.source_url,
        snippet=f"{note.title} (issued {note.issued_raw}). {note.body}".strip(),
        title=note.title or "Notify NYC notification",
        kind="DATA",
        valid_as_of=note.issued or note.issued_raw,
        provenance=provenance,
    )


def _recent_citation(ctx: ToolContext, note: RecentNote) -> str:
    return _register_recent_citation(ctx.citations, note)


def _recent_block(note: RecentNote, cite: str) -> str:
    parts = [f"- {note.title or 'Notify NYC notification'} (issued {note.issued_raw or 'unknown'}) "
             f"{{cite:{cite}}}"]
    if note.body:
        parts.append(f"  {note.body}")
    return "\n".join(parts)


def _recent_awareness(
    feed,
    today: date,
    *,
    citations: CitationRegistry | None = None,
) -> str:
    cutoff = today - timedelta(days=6)
    notes = [
        note for note in feed.notes
        if note.issued
        and cutoff <= datetime.fromisoformat(note.issued).date() <= today
    ]
    if not feed.confirmed or not notes:
        return ""
    lines = [
        "Notify NYC notifications from the last seven days (NYC Emergency Management), full "
        "text below. They are exact cached messages and a relevance hint, not answer evidence. "
        "Before mentioning any notice, call `check_notify_nyc` so the current matching source is "
        "registered. Judge each cached notice by its meaning:",
        "- A notice about immediate personal safety: surface it proactively even when you do not "
        "know where the resident is — state its area and let them judge (\"if you're in the "
        "Bronx...\"). Never withhold a safety notice because the resident hasn't shared a location.",
        "- A narrow, low-stakes notice (one street, one facility): mention it only when the "
        "resident's known location or question overlaps it.",
        "- Compare any end time a notice states against the current time; do not urge action on "
        "one that has already ended.",
    ]
    omitted = 0
    for index, note in enumerate(notes):
        block = f"- {note.issued_raw}: {note.title}"
        if citations is not None:
            block += f" {{cite:{_register_recent_citation(citations, note)}}}"
        if note.body:
            block += f"\n  {note.body}"
        if len("\n".join((*lines, block))) > _AWARENESS_MAX_CHARS:
            omitted = len(notes) - index
            break
        lines.append(block)
    if omitted:
        lines.append(
            f"{omitted} older notification(s) remain cached but are omitted from this prompt. "
            "Call `check_notify_nyc` if the resident asks for current advisory detail."
        )
    return "\n".join(lines)


async def current_awareness(
    citations: CitationRegistry | None = None,
) -> str:
    global _awareness_cache

    now = monotonic()
    if _awareness_cache is not None and now - _awareness_cache[0] < _AWARENESS_TTL_S:
        notes = sorted(
            _awareness_cache[1].values(), key=lambda note: note.issued, reverse=True,
        )
        return _recent_awareness(
            RecentFeed(confirmed=bool(notes), notes=notes),
            datetime.now(NYC_TZ).date(),
            citations=citations,
        )
    async with _awareness_lock:
        now = monotonic()
        if _awareness_cache is not None and now - _awareness_cache[0] < _AWARENESS_TTL_S:
            notes = sorted(
                _awareness_cache[1].values(), key=lambda note: note.issued, reverse=True,
            )
            return _recent_awareness(
                RecentFeed(confirmed=bool(notes), notes=notes),
                datetime.now(NYC_TZ).date(),
                citations=citations,
            )
        feed = await fetch_recent_advisories()
        today = datetime.now(NYC_TZ).date()
        cached = dict(_awareness_cache[1]) if _awareness_cache else {}
        if feed.confirmed:
            cached.update((note.guid, note) for note in feed.notes)
        cutoff = today - timedelta(days=6)
        cached = {
            guid: note for guid, note in cached.items()
            if note.issued and cutoff <= datetime.fromisoformat(note.issued).date() <= today
        }
        if feed.confirmed or cached:
            _awareness_cache = (monotonic(), cached)
        notes = sorted(cached.values(), key=lambda note: note.issued, reverse=True)
        awareness = _recent_awareness(
            RecentFeed(confirmed=bool(notes), notes=notes),
            today,
            citations=citations,
        )
        if not feed.confirmed and awareness:
            return "Notify NYC refresh failed; showing unexpired cached messages.\n" + awareness
        return awareness


def _render_cap(ctx: ToolContext, advisories: list[Advisory], near: str) -> str:
    """The structured CAP report: each active advisory with severity + 'in effect until', cited."""
    lines = [
        "Active NYC advisories from the Notify NYC feed (NYC Emergency Management). Use only "
        "notices relevant to the resident's question. For a broad advisory question, report these "
        "notices, cite each one, and state its 'in effect until' time. For a question about one "
        "alert type, do not mention or cite unrelated notices. If none match, say the feed did not "
        f"return a matching notice and provide {NOTIFY_NYC_URL} for a direct check:",
    ]
    if near:
        lines.append(
            f"(User asked about '{near}'. The feed's geography is usually citywide, so these are "
            f"returned for individual area and time comparison rather than suppressed by the "
            f"location hint.)"
        )
    for advisory in advisories:
        cite = _advisory_citation(ctx, advisory)
        lines.append(_advisory_block(advisory, cite))
    lines.append(
        "Limits: this is the public Notify NYC feed; its area is often citywide even for a local "
        "event, and it can lag the SMS/email alerts by a few minutes. Never state a severity or "
        "expiry the feed didn't give. If a HEAT advisory is active, also offer cooling centers; for "
        "AIR QUALITY, pass along the advisory's sensitive-groups guidance; for a CLOSURE/transport "
        "advisory, point to transit."
    )
    lines.append(PLAN_RELEVANCE)
    return "\n".join(lines)


def _render_recent(ctx: ToolContext, notes: list[RecentNote], near: str) -> str:
    """The real-time FALLBACK report used when the CAP feed is empty/degraded: the city's own live
    Notify NYC notifications, newest first, cited to the resolvable endpoint. Thinner than CAP (no
    machine-readable severity/expiry), so we show each note's ISSUE time instead of an invented one."""
    lines = [
        "The Notify NYC CAP feed was empty or unreachable, so these are the CITY'S OWN live Notify "
        f"NYC notifications (newest first) from {NOTIFY_NYC_URL}. This live source gives "
        "an ISSUE TIME, not a machine-readable 'in effect until', so state each notification's issue "
        "time and let the user judge how current it is. Use only notices relevant to the resident's "
        "question. For a question about one alert type, do not mention or cite unrelated notices. "
        "If none match, say the feed did not return a matching notice and provide the Notify NYC "
        "URL above for a direct check. Do NOT invent a severity or an expiry:",
    ]
    if near:
        lines.append(
            f"(User asked about '{near}'. Notify NYC is usually citywide, so these are returned for "
            f"individual area and time comparison rather than suppressed by the location hint.)"
        )
    for note in notes:
        cite = _recent_citation(ctx, note)
        lines.append(_recent_block(note, cite))
    lines.append(
        "Limits: this live fallback lists recent notifications without a structured expiry, so "
        "confirm currency by the issue time, and for the fullest picture also point the user to "
        f"{NOTIFY_NYC_URL} and 311. If a HEAT notification is active, also offer cooling centers; for "
        "FLOODING, pass along any safe-location or road-safety guidance in the notification text. For "
        "a life-threatening emergency, tell the user to call 911 right away."
    )
    lines.append(PLAN_RELEVANCE)
    return "\n".join(lines)


def _render_recent_additions(ctx: ToolContext, notes: list[RecentNote]) -> str:
    lines = [
        "Additional current notifications from the City's live Notify NYC messages endpoint. "
        "These have an issue time but no machine-readable expiry:",
    ]
    for note in notes:
        lines.append(_recent_block(note, _recent_citation(ctx, note)))
    lines.append(PLAN_RELEVANCE)
    return "\n".join(lines)


# F080 residual: the model re-fetches advisories it already delivered and re-briefs them.
# When every current item was already cited earlier in the conversation, the tool answers
# with this marker instead of the payload, so there is nothing to re-brief.
_ALREADY_SHARED = (
    "Nothing new: the active advisories are unchanged since earlier in this conversation "
    "({titles}). Do not re-brief them; answer the resident's actual request, referring back "
    "briefly only if one is directly relevant. If the resident explicitly asks to see the "
    "details again, call check_notify_nyc with full_text=true."
)
_STILL_ACTIVE = (
    "Also still active and already shared earlier in this conversation (do not re-brief): "
    "{titles}"
)


def _norm_title(title: str) -> str:
    return (title or "").strip().casefold()


async def _current_sources(
    ctx: ToolContext,
    *,
    now: datetime,
    lang: str | None,
):
    return await asyncio.gather(
        active_advisories(ctx.http, now=now, lang=lang),
        fetch_recent_advisories(ctx.http, lang=lang),
    )


def _candidate_listing(
    advisories: list[Advisory],
    notes: list[RecentNote],
) -> str:
    lines = [
        "Current Notify NYC candidates. These are source records for semantic selection, not "
        "answer citations. Select by meaning, then call `check_notify_nyc` with only the relevant "
        "notice_ids. Call it with an empty notice_ids list when none match:"
    ]
    for advisory in advisories:
        lines.append(
            f"- {advisory.headline or advisory.event or 'NYC advisory'} "
            f"(notice_id={_cap_notice_id(advisory)})\n"
            f"  Event: {advisory.event or 'unknown'}; category: "
            f"{advisory.category or 'unknown'}; sent: {advisory.sent or 'unknown'}; expires: "
            f"{advisory.expires or 'unknown'}; area: {advisory.area_desc or 'unknown'}"
        )
        if advisory.description:
            lines.append(f"  {advisory.description}")
        if advisory.instruction:
            lines.append(f"  Instruction: {advisory.instruction}")
    for note in notes:
        lines.append(
            f"- {note.title or 'Notify NYC notification'} (notice_id={note.guid})\n"
            f"  Issued: {note.issued_raw or 'unknown'}"
        )
        if note.body:
            lines.append(f"  {note.body}")
    return "\n".join(lines)


async def _list_handler(args: dict, ctx: ToolContext) -> str:
    lang = (args.get("lang") or "").strip() or None
    now = datetime.now(timezone.utc)
    feed, recent = await _current_sources(ctx, now=now, lang=lang)
    cap_titles = {
        _norm_title(advisory.headline or advisory.event)
        for advisory in feed.advisories
    }
    recent_notes = [
        note for note in recent.notes if _norm_title(note.title) not in cap_titles
    ]
    if feed.advisories or recent_notes:
        return _candidate_listing(feed.advisories, recent_notes)
    if feed.confirmed:
        return NO_ACTIVE
    return COULD_NOT_CONFIRM


def _render_recent_delta(
    ctx: ToolContext,
    notes: list[RecentNote],
    delivered: frozenset,
    render: Callable[[ToolContext, list[RecentNote]], str],
) -> str:
    new_notes = [note for note in notes if _norm_title(note.title) not in delivered]
    old_notes = sorted(
        (note for note in notes if note not in new_notes), key=lambda note: note.title
    )
    old_titles = "; ".join(
        f"{note.title} {{cite:{_recent_citation(ctx, note)}}}" for note in old_notes
    )
    if delivered and not new_notes:
        return _ALREADY_SHARED.format(titles=old_titles)
    rendered = render(ctx, new_notes)
    if old_notes:
        rendered = f"{rendered}\n\n{_STILL_ACTIVE.format(titles=old_titles)}"
    return rendered


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    lang = (args.get("lang") or "").strip() or None
    delivered = frozenset() if args.get("full_text") else ctx.delivered_notify_titles
    now = datetime.now(timezone.utc)

    # TIERED, fail-safe sourcing (neither call ever crashes; both carry a `confirmed` flag):
    #   1. CAP/Everbridge feed (structured: severity + 'in effect until'). Primary WHEN it works.
    #   2. The city's own live "recent messages" endpoint. Real-time fallback WHEN CAP is degraded
    #      (which is the current reality: the CAP feed has been publishing empty during emergencies).
    #   3. Fail safe: if NEITHER source can be confirmed, we NEVER say "no advisories"; we say we
    #      could not confirm and route to the official live source + 311 (+ 911 for a life-threat).
    # `lang` surfaces the official city translation of a CAP advisory where the feed carries it.
    feed, recent = await _current_sources(ctx, now=now, lang=lang)
    # F061: no area filter that parses notice prose — the feed spells areas however it likes
    # ("BK/SI/MN/QN", "parts of NYC"), so every active notice returns and the model judges
    # relevance from the full cited text.
    all_cap_advisories = feed.advisories
    all_recent_notes = recent.notes
    cap_advisories = all_cap_advisories
    recent_notes = all_recent_notes
    if "notice_ids" in args:
        selected_notice_ids = list(dict.fromkeys(args.get("notice_ids") or ()))
        query = (args.get("query") or "").strip()
        if not selected_notice_ids:
            cite = _selection_receipt_citation(
                ctx,
                query,
                selected_notice_ids,
                cap_confirmed=feed.confirmed,
                recent_confirmed=recent.confirmed,
                cap_records=len(all_cap_advisories),
                recent_records=len(all_recent_notes),
                observed_at=now,
            )
            return (
                f"No current notice was selected for {query or 'the resident request'} "
                f"{{cite:{cite}}}. This does not establish a citywide all-clear. Check Notify NYC "
                "using the attached Notify NYC source link or call 311 for confirmation."
            )
        selected = set(selected_notice_ids)
        cap_advisories = [
            advisory for advisory in all_cap_advisories
            if _cap_notice_id(advisory) in selected
        ]
        recent_notes = [note for note in all_recent_notes if note.guid in selected]
        found = {
            *(_cap_notice_id(advisory) for advisory in cap_advisories),
            *(note.guid for note in recent_notes),
        }
        if not found:
            cite = _selection_receipt_citation(
                ctx,
                query,
                selected_notice_ids,
                cap_confirmed=feed.confirmed,
                recent_confirmed=recent.confirmed,
                cap_records=len(all_cap_advisories),
                recent_records=len(all_recent_notes),
                observed_at=now,
            )
            return (
                f"The selected Notify NYC notice IDs were no longer present {{cite:{cite}}}. "
                f"Check {NOTIFY_NYC_URL} directly or call 311 for confirmation."
            )

    if feed.confirmed and cap_advisories:
        cap_titles = {
            _norm_title(advisory.headline or advisory.event)
            for advisory in cap_advisories
        }
        additions = [
            note for note in recent_notes
            if _norm_title(note.title) not in cap_titles
        ] if recent.confirmed else []
        # F080: split current items into new vs already delivered this conversation.
        # F083: already-shared mentions each carry their OWN citation, registered fresh, so a
        # legitimate refer-back binds to the right source; a citation-free marker pushed the
        # model to pin remembered facts on whatever fresh cite was at hand (observed live).
        old_caps = [a for a in cap_advisories
                    if _norm_title(a.headline or a.event) in delivered]
        new_caps = [a for a in cap_advisories if a not in old_caps]
        new_additions = [n for n in additions if _norm_title(n.title) not in delivered]

        def old_mentions() -> list[str]:
            # Registered AFTER any new items so citation ids follow reading order.
            return sorted(
                [f"{a.headline or a.event} {{cite:{_advisory_citation(ctx, a)}}}"
                 for a in old_caps]
                + [f"{n.title} {{cite:{_recent_citation(ctx, n)}}}" for n in additions
                   if n not in new_additions]
            )

        if delivered and not new_caps and not new_additions:
            return _ALREADY_SHARED.format(titles="; ".join(old_mentions()))
        parts = []
        if new_caps:
            parts.append(_render_cap(ctx, new_caps, near))
        if new_additions:
            parts.append(_render_recent_additions(ctx, new_additions))
        mentions = old_mentions()
        if mentions:
            parts.append(_STILL_ACTIVE.format(titles="; ".join(mentions)))
        return "\n\n".join(parts)
    if feed.confirmed:
        if recent.confirmed and recent_notes:
            return _render_recent_delta(
                ctx, recent_notes, delivered, _render_recent_additions
            )
        # F067: our own forced game-day/prep check finding nothing is not news — return
        # nothing so there is no null result for the model to narrate as an "update".
        # A resident who actually asked gets the plain NO_ACTIVE answer.
        if args.get("incidental"):
            return ""
        return NO_ACTIVE

    # CAP feed DEGRADED (unreachable / empty body / unreadable), so do NOT trust its emptiness as an
    # all-clear. Consult the city's live notifications before ever telling the user there are none.
    if recent.confirmed and recent_notes:
        return _render_recent_delta(
            ctx,
            recent_notes,
            delivered,
            lambda context, notes: _render_recent(context, notes, near),
        )
    return COULD_NOT_CONFIRM                                    # both sources degraded -> fail safe


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="check_notify_nyc",
            description=(
                "Report the NYC emergency advisories currently in effect, grounded in the live "
                "Notify NYC / NYC Emergency Management feed (extreme heat, air quality, boil-water, "
                "beach/pool closures, transit disruptions, and more). Use this for 'are there any "
                "advisories/alerts right now', 'is it safe outside today', a heat or air-quality "
                "warning, 'is <beach> open', or a boil-water question. Optional `near` = the user's "
                "NYC address/neighborhood (the feed's geography is usually citywide, so results are "
                "NOT filtered by it). Returns each active advisory with its headline, severity, "
                "event, 'in effect until <expires>', and area, every one carrying a DATA citation. "
                "If none are active it says so plainly, it NEVER invents an advisory, severity, or "
                "expiry."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {
                        "type": "string",
                        "description": "Optional NYC address/neighborhood for context (results are "
                        "not geo-filtered, the feed is usually citywide).",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Optional language NAME for the advisory text (e.g. 'Spanish', "
                        "'Chinese'), pass the language the user is writing in. The feed carries "
                        "official city translations for ~12 languages; defaults to English, and "
                        "uses the English feed only when the requested language is absent.",
                    },
                    "notice_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "One stable notice ID from `list_notify_nyc`.",
                        },
                        "description": "Stable notice IDs returned by `list_notify_nyc`. For a "
                        "specific-alert question, pass only the semantically relevant IDs, or an "
                        "empty list when none match. Omit only for a broad request for all current "
                        "advisories.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The resident's requested alert topic, in their language. "
                        "Required with notice_ids so an empty or changed selection has an honest "
                        "search receipt. This documents the selection and does not perform string "
                        "matching.",
                    },
                    "full_text": {
                        "type": "boolean",
                        "description": "Set true ONLY when the resident explicitly asks to see "
                        "again details you already shared in this conversation; otherwise "
                        "already-shared advisories return as a compact unchanged marker.",
                    },
                },
            },
            handler=_handler,
            open_world=True,  # hits the live Notify NYC / Everbridge feed
        ),
        Tool(
            name="list_notify_nyc",
            description=(
                "List the compact current Notify NYC candidates, with stable notice IDs and "
                "source text but no answer citations. Use this first for a question about one "
                "kind of alert, select relevant records by meaning, then call check_notify_nyc "
                "with those notice_ids. Pass `lang` so Notify NYC supplies its official "
                "translation where available."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lang": {
                        "type": "string",
                        "description": "Optional language NAME or code matching the resident's "
                        "language. Uses that official language feed when present, or the English "
                        "feed when the requested language is absent.",
                    },
                },
            },
            handler=_list_handler,
            open_world=True,
        ),
    ]
