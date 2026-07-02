"""advisories module tool: `nyc_advisories`, grounded in the live Notify NYC feed.

Data source: the public Notify NYC / NYC Emergency Management Everbridge RSS feed of CAP alerts
(see heynyc/core/tools/notify_nyc.py). On demand, we fetch the feed, keep the advisories still in
effect at `now`, and report each one — headline, severity, event, "in effect until <expires>", and
areaDesc — with a DATA citation registered to the advisory's resolvable CAP XML url. When nothing
is active we abstain cleanly rather than manufacture an alert.

Honest limitations (enforced in the manifest prompt too): the feed's geography is often citywide
even for a local event, so we do NOT filter out citywide alerts on a `near` hint; and the feed can
lag the SMS/email alerts by minutes. We never invent an advisory, severity, or expiry.
"""
from __future__ import annotations

from datetime import datetime, timezone

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.notify_nyc import Advisory, active_advisories

OFFICIAL = "Notify NYC (nyc.gov/notifynyc) or call 311"


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


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    now = datetime.now(timezone.utc)

    # active_advisories is resilient (returns [] on any network/parse error) — never crashes.
    advisories = await active_advisories(ctx.http, now=now)

    if not advisories:
        return (
            "No active Notify NYC advisories came back from the feed right now. Do NOT invent one — "
            "tell the user there are no active Notify NYC advisories at the moment (this is the "
            f"public Notify NYC feed, not the whole picture) and point them to {OFFICIAL}. "
            "Offer to check again."
        )

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
                },
            },
            handler=_handler,
            open_world=True,  # hits the live Notify NYC / Everbridge feed
        )
    ]
