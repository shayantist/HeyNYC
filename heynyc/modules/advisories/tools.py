"""advisories module tool: `nyc_advisories`, grounded in the live Notify NYC feed.

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

from datetime import datetime, timezone

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.notify_nyc import (
    Advisory,
    RecentNote,
    active_advisories,
    fetch_recent_advisories,
)

OFFICIAL = "Notify NYC (nyc.gov/notifynyc) or call 311"

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
    "an all-clear. Tell them plainly that we could not confirm active advisories at the moment and "
    "that an emergency may still be in effect, then send them to the official live source, Notify NYC "
    "at nyc.gov/notifynyc, and to 311 for current advisories. For a life-threatening emergency, tell "
    "them to call 911 right away. Offer to check again."
)


def _advisory_snapshot(advisory: Advisory) -> dict:
    """The exact fields cited, hashed into the DATA citation for reproducibility/integrity."""
    return {
        "identifier": advisory.guid,
        "headline": advisory.headline,
        "event": advisory.event,
        "category": advisory.category,
        "severity": advisory.severity,
        "urgency": advisory.urgency,
        "sent": advisory.sent,
        "expires": advisory.expires,
        "areaDesc": advisory.area_desc,
    }


def _advisory_citation(ctx: ToolContext, advisory: Advisory) -> str:
    """Register a DATA citation grounded in the advisory's resolvable CAP XML (re-fetchable + hashed).
    `valid_as_of` is the advisory's own `sent` time — temporal provenance, never fetch time."""
    provenance = data_provenance(
        _advisory_snapshot(advisory), record_id=advisory.guid, field_pointer="/"
    )
    return ctx.citations.register(
        advisory.source_url,
        snippet=f"{advisory.headline or advisory.event} — in effect until {advisory.expires}",
        title="Notify NYC / NYC Emergency Management",
        kind="DATA",
        valid_as_of=advisory.sent,
        provenance=provenance,
    )


def _advisory_block(advisory: Advisory, cite: str) -> str:
    headline = advisory.headline or advisory.event or "NYC advisory"
    severity = advisory.severity or "Unknown"
    parts = [
        f"- {headline} [{severity}] — {advisory.event or 'advisory'}, "
        f"in effect until {advisory.expires} {{cite:{cite}}}"
    ]
    parts.append(f"  Area (per feed): {advisory.area_desc or 'NYC'}")
    if advisory.category:
        parts.append(f"  Category: {advisory.category}")
    parts.append(f"  As of: {advisory.sent or 'unknown'}")
    return "\n".join(parts)


def _recent_citation(ctx: ToolContext, note: RecentNote) -> str:
    """DATA citation for a live "recent messages" note, grounded in the resolvable Notify NYC endpoint.
    `valid_as_of` is the notification's own ISSUE time (temporal provenance), never fetch time."""
    snapshot = {"title": note.title, "body": note.body, "pubDate": note.issued_raw}
    provenance = data_provenance(snapshot, record_id=note.guid, field_pointer="/")
    return ctx.citations.register(
        note.source_url,
        snippet=f"{note.title} (issued {note.issued_raw})",
        title="Notify NYC (live recent messages)",
        kind="DATA",
        valid_as_of=note.issued or note.issued_raw,
        provenance=provenance,
    )


def _recent_block(note: RecentNote, cite: str) -> str:
    parts = [f"- {note.title or 'Notify NYC notification'} (issued {note.issued_raw or 'unknown'}) "
             f"{{cite:{cite}}}"]
    if note.body:
        parts.append(f"  {note.body}")
    return "\n".join(parts)


def _render_cap(ctx: ToolContext, advisories: list[Advisory], near: str) -> str:
    """The structured CAP report: each active advisory with severity + 'in effect until', cited."""
    lines = [
        "Active NYC advisories from the Notify NYC feed (NYC Emergency Management) — report ONLY "
        "these, cite each, and state each one's 'in effect until' time:",
    ]
    if near:
        lines.append(
            f"(User asked about '{near}'. The feed's geography is usually citywide, so these are "
            f"listed as-is — do not filter them out for a specific location.)"
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
    return "\n".join(lines)


def _render_recent(ctx: ToolContext, notes: list[RecentNote], near: str) -> str:
    """The real-time FALLBACK report used when the CAP feed is empty/degraded: the city's own live
    Notify NYC notifications, newest first, cited to the resolvable endpoint. Thinner than CAP (no
    machine-readable severity/expiry), so we show each note's ISSUE time instead of an invented one."""
    lines = [
        "The Notify NYC CAP feed was empty or unreachable, so these are the CITY'S OWN live Notify "
        "NYC notifications (newest first) from the nyc.gov/notifynyc source. This live source gives "
        "an ISSUE TIME, not a machine-readable 'in effect until', so state each notification's issue "
        "time and let the user judge how current it is. Report ONLY these, cite each, and do NOT "
        "invent a severity or an expiry:",
    ]
    if near:
        lines.append(
            f"(User asked about '{near}'. Notify NYC is usually citywide, so list these as-is and do "
            f"not filter them out for a specific location.)"
        )
    for note in notes:
        cite = _recent_citation(ctx, note)
        lines.append(_recent_block(note, cite))
    lines.append(
        "Limits: this live fallback lists recent notifications without a structured expiry, so "
        "confirm currency by the issue time, and for the fullest picture also point the user to "
        "nyc.gov/notifynyc and 311. If a HEAT notification is active, also offer cooling centers; for "
        "FLOODING, pass along any safe-location or road-safety guidance in the notification text. For "
        "a life-threatening emergency, tell the user to call 911 right away."
    )
    return "\n".join(lines)


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    lang = (args.get("lang") or "").strip() or None
    now = datetime.now(timezone.utc)

    # TIERED, fail-safe sourcing (neither call ever crashes; both carry a `confirmed` flag):
    #   1. CAP/Everbridge feed (structured: severity + 'in effect until'). Primary WHEN it works.
    #   2. The city's own live "recent messages" endpoint. Real-time fallback WHEN CAP is degraded
    #      (which is the current reality: the CAP feed has been publishing empty during emergencies).
    #   3. Fail safe: if NEITHER source can be confirmed, we NEVER say "no advisories"; we say we
    #      could not confirm and route to the official live source + 311 (+ 911 for a life-threat).
    # `lang` surfaces the official city translation of a CAP advisory where the feed carries it.
    feed = await active_advisories(ctx.http, now=now, lang=lang)

    if feed.confirmed and feed.advisories:
        return _render_cap(ctx, feed.advisories, near)          # best case: structured + active
    if feed.confirmed:
        return NO_ACTIVE                                        # working CAP feed, genuinely nothing active

    # CAP feed DEGRADED (unreachable / empty body / unreadable), so do NOT trust its emptiness as an
    # all-clear. Consult the city's live notifications before ever telling the user there are none.
    recent = await fetch_recent_advisories(ctx.http)
    if recent.confirmed and recent.notes:
        return _render_recent(ctx, recent.notes, near)          # real-time fallback (today's alerts)
    return COULD_NOT_CONFIRM                                    # both sources degraded -> fail safe


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="nyc_advisories",
            description=(
                "Report the NYC emergency advisories currently in effect, grounded in the live "
                "Notify NYC / NYC Emergency Management feed (extreme heat, air quality, boil-water, "
                "beach/pool closures, transit disruptions, and more). Use this for 'are there any "
                "advisories/alerts right now', 'is it safe outside today', a heat or air-quality "
                "warning, 'is <beach> open', or a boil-water question. Optional `near` = the user's "
                "NYC address/neighborhood (the feed's geography is usually citywide, so results are "
                "NOT filtered by it). Returns each active advisory with its headline, severity, "
                "event, 'in effect until <expires>', and area, every one carrying a DATA citation. "
                "If none are active it says so plainly — it NEVER invents an advisory, severity, or "
                "expiry."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {
                        "type": "string",
                        "description": "Optional NYC address/neighborhood for context (results are "
                        "not geo-filtered — the feed is usually citywide).",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Optional language NAME for the advisory text (e.g. 'Spanish', "
                        "'Chinese') — pass the language the user is writing in. The feed carries "
                        "official city translations for ~12 languages; defaults to English, and "
                        "falls back to English for any alert with no variant in that language.",
                    },
                },
            },
            handler=_handler,
            open_world=True,  # hits the live Notify NYC / Everbridge feed
        )
    ]
