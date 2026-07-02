"""housing module tool: `hpd_building_lookup` — a building's OPEN HPD complaints + violations.

Grounded in two NYC Open Data (Socrata) datasets, addressed by the building's tax-lot (BBL):

  - Housing Maintenance Code Complaints  (ygpa-z7cr) — filtered to complaint_status='OPEN',
    summarized by major_category (HEAT/HOT WATER called out).
  - Housing Maintenance Code Violations  (wvxf-dwi5) — filtered to violationstatus='Open',
    summarized by class (class C = immediately hazardous, e.g. no-heat-in-season, called out).

Flow: geocode the address (reusing the shared geocoder, which now carries the PAD `bbl`), then
query both datasets by BBL. A specific street address is REQUIRED — a bare ZIP or a neighborhood
has no building BBL, so the tool abstains and asks for a street address rather than guess a
building. Each dataset query is a resolvable, filtered DATA citation (re-fetch → verify). We only
report what the data shows; empty results are stated plainly (no open records), never spun into
"the building is fine." Filing a NEW no-heat/no-hot-water complaint is routed to 311.
"""
from __future__ import annotations

import collections
from urllib.parse import quote

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset
from heynyc.core.tools.geo import geocode

COMPLAINTS_ID = "ygpa-z7cr"   # HPD Housing Maintenance Code Complaints
VIOLATIONS_ID = "wvxf-dwi5"   # HPD Housing Maintenance Code Violations
OFFICIAL = "NYC311 (call 311 or nyc.gov/311)"


def _filtered_url(dataset_id: str, where: str) -> str:
    """A resolvable, filtered Socrata permalink: /resource/{4x4}.json?$where=<encoded>. Re-fetching
    it returns exactly the rows this citation is grounded in (auditable)."""
    return f"{dataset_url(dataset_id)}?$where={quote(where)}"


def _iso_date(value) -> str:
    """A Socrata datetime ('2026-06-30T06:50:52.000') → 'YYYY-MM-DD'; '' if blank/odd."""
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 else ""


def _most_recent(records: list[dict], field: str) -> str:
    """The latest date in `field` across the rows (ISO strings sort lexically), 'YYYY-MM-DD'."""
    dates = [_iso_date(r.get(field)) for r in records]
    dates = [d for d in dates if d]
    return max(dates) if dates else ""


def _counts(records: list[dict], field: str) -> collections.Counter:
    return collections.Counter(str(r.get(field) or "?").strip().upper() for r in records)


def _summary_line(counts: collections.Counter) -> str:
    """'CLASS: n' descending by count, for a compact grounded breakdown."""
    return ", ".join(f"{name}: {n}" for name, n in counts.most_common())


def _register(ctx: ToolContext, dataset_id: str, where: str, *, title: str, snippet: str,
              snapshot: dict, valid_as_of: str) -> str:
    prov = data_provenance(snapshot, record_id=snapshot.get("bbl", ""), field_pointer="/")
    return ctx.citations.register(
        _filtered_url(dataset_id, where),
        snippet=snippet,
        title=title,
        kind="DATA",
        valid_as_of=valid_as_of,
        provenance=prov,
    )


async def _handler(args: dict, ctx: ToolContext) -> str:
    address = (args.get("address") or "").strip()
    if not address:
        return ("Ask the user for a specific NYC street address (building number + street) before "
                "looking up a building — never guess a building.")

    origin = await geocode(address, client=ctx.http)
    if origin is None or not origin.bbl:
        # A bare ZIP, a neighborhood, or a non-NYC place has no building BBL — the lookup is
        # building-level, so abstain and ask for a street address rather than guess a building.
        return (f"I couldn't tie '{address}' to a specific NYC building, so I can't pull its HPD "
                f"record. Ask the user for a specific NYC street address (building number + street) "
                f"— a ZIP or neighborhood alone doesn't identify a building. Don't guess a building.")

    bbl = origin.bbl
    # BBL is 10 chars: 1 borough digit, 5 block digits, 4 lot digits. The complaints dataset keys on
    # the full BBL; the violations dataset keys on numeric boro/block/lot (leading zeros stripped).
    boro, block, lot = bbl[0], str(int(bbl[1:6])), str(int(bbl[6:10]))

    complaints_where = f"bbl='{bbl}' AND complaint_status='OPEN'"
    violations_where = (f"boroid='{boro}' AND block='{block}' AND lot='{lot}' "
                        f"AND violationstatus='Open'")
    try:
        complaints = await query_dataset(COMPLAINTS_ID, where=complaints_where, limit=1000,
                                         client=ctx.http)
        violations = await query_dataset(VIOLATIONS_ID, where=violations_where, limit=1000,
                                         client=ctx.http)
    except httpx.HTTPError:
        return (f"I couldn't reach the city's HPD data right now — don't guess whether the building "
                f"has complaints or violations. Point the user to {OFFICIAL} and hpdonline.nyc.gov.")

    cat_counts = _counts(complaints, "major_category")
    heat_complaints = cat_counts.get("HEAT/HOT WATER", 0)
    complaints_recent = _most_recent(complaints, "received_date")

    class_counts = _counts(violations, "class")
    class_c = class_counts.get("C", 0)
    violations_recent = _most_recent(violations, "novissueddate")

    if not complaints and not violations:
        # Say so plainly — DON'T imply the building is trouble-free beyond what the data covers.
        cite_c = _register(ctx, COMPLAINTS_ID, complaints_where,
                           title="HPD Housing Maintenance Code Complaints (open)",
                           snippet=f"BBL {bbl}: 0 open HPD complaints",
                           snapshot={"bbl": bbl, "open_complaints": 0}, valid_as_of="")
        cite_v = _register(ctx, VIOLATIONS_ID, violations_where,
                           title="HPD Housing Maintenance Code Violations (open)",
                           snippet=f"BBL {bbl}: 0 open HPD violations",
                           snapshot={"bbl": bbl, "open_violations": 0}, valid_as_of="")
        return (
            f"Building: {origin.label} (BBL {bbl})\n"
            f"No OPEN HPD complaints {{cite:{cite_c}}} or OPEN HPD violations {{cite:{cite_v}}} are on "
            f"record for this building right now. That only reflects what tenants have reported to HPD "
            f"and what HPD has cited — it doesn't guarantee there are no problems. If the user has a "
            f"heat, hot-water, or repair issue, they can still file a complaint through {OFFICIAL}."
        )

    cite_c = _register(
        ctx, COMPLAINTS_ID, complaints_where,
        title="HPD Housing Maintenance Code Complaints (open)",
        snippet=f"BBL {bbl}: {len(complaints)} open HPD complaints ({heat_complaints} heat/hot-water)",
        snapshot={"bbl": bbl, "open_complaints": len(complaints), "heat_hot_water": heat_complaints,
                  "by_major_category": dict(cat_counts)},
        valid_as_of=complaints_recent,
    )
    cite_v = _register(
        ctx, VIOLATIONS_ID, violations_where,
        title="HPD Housing Maintenance Code Violations (open)",
        snippet=f"BBL {bbl}: {len(violations)} open HPD violations ({class_c} class C)",
        snapshot={"bbl": bbl, "open_violations": len(violations), "class_c": class_c,
                  "by_class": dict(class_counts)},
        valid_as_of=violations_recent,
    )

    lines = [f"Building: {origin.label} (BBL {bbl})", "Grounded in NYC HPD open data — report only these:"]

    lines.append(f"- Open HPD complaints: {len(complaints)} total"
                 + (f", including {heat_complaints} HEAT/HOT WATER" if heat_complaints else "")
                 + f" {{cite:{cite_c}}}")
    if complaints:
        lines.append(f"  By category: {_summary_line(cat_counts)}")
    if complaints_recent:
        lines.append(f"  Most recent complaint received: {complaints_recent}")

    lines.append(f"- Open HPD violations: {len(violations)} total"
                 + (f", including {class_c} class C (immediately hazardous — includes no-heat in season)"
                    if class_c else "")
                 + f" {{cite:{cite_v}}}")
    if violations:
        lines.append(f"  By class: {_summary_line(class_counts)}"
                     " (A = non-hazardous, B = hazardous, C = immediately hazardous)")
    if violations_recent:
        lines.append(f"  Most recent violation issued: {violations_recent}")

    lines.append(
        f"Tell the tenant these are the building's OPEN records only (issues already reported/cited). "
        f"To report a NEW no-heat / no-hot-water or repair problem, they file through {OFFICIAL}. "
        f"Don't invent counts, dates, or outcomes beyond what's cited above."
    )
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="hpd_building_lookup",
            description=(
                "Look up a specific NYC building's OPEN HPD complaints and violations (heat/hot-water, "
                "safety, unsanitary conditions, etc.), grounded in NYC Open Data and cited. Pass "
                "`address` = a specific NYC STREET address (building number + street). Returns counts "
                "by category/class — calling out HEAT/HOT WATER complaints and class C (immediately "
                "hazardous) violations — plus the most recent dates. If the address is only a ZIP or a "
                "neighborhood (no building BBL), the tool abstains and asks for a street address; it "
                "never guesses a building. Empty results are reported as 'no open records', not "
                "'problem-free'. Use for 'does my building have heat/violations, is my landlord in "
                "trouble' — to FILE a new complaint, route the user to 311."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "A specific NYC street address (building number + street), "
                                       "e.g. '617 Courtlandt Ave, Bronx'.",
                    },
                },
                "required": ["address"],
            },
            handler=_handler,
            open_world=True,  # hits the live Socrata HPD datasets + geocoder
        )
    ]
