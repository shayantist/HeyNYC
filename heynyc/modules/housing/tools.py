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
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset
from heynyc.core.tools.geo import geocode

COMPLAINTS_ID = "ygpa-z7cr"   # HPD Housing Maintenance Code Complaints
VIOLATIONS_ID = "wvxf-dwi5"   # HPD Housing Maintenance Code Violations
LITIGATIONS_ID = "59kj-x8nc"  # HPD Housing Litigations (housing-court cases, keyed by BBL)
OFFICIAL = "NYC311 (call 311 or nyc.gov/311)"

# ygpa-z7cr codes OPEN heat complaints under TWO major_category values: 'HEAT/HOT WATER' and the
# separate 'HEATING' (minor categories HEAT RELATED / RADIATOR / HEAT-PLANT / SPACE HEATER, all
# genuinely heat). Counting only 'HEAT/HOT WATER' undercounts heat, so the call-out spans both.
HEAT_CATEGORIES = ("HEAT/HOT WATER", "HEATING")
HEAT_CASETYPE = "Heat and Hot Water"   # the 59kj-x8nc casetype for a housing-court heat case
HARASSMENT_FINDINGS = ("AFTER INQUEST", "AFTER TRIAL")  # findingofharassment values = a positive finding


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
    heat_complaints = sum(cat_counts.get(c, 0) for c in HEAT_CATEGORIES)
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
                 + (f", including {heat_complaints} heat/hot-water" if heat_complaints else "")
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


async def _litigation_handler(args: dict, ctx: ToolContext) -> str:
    """A building's HPD Housing Litigations (59kj-x8nc): whether HPD or a tenant has taken the
    landlord to Housing Court, keyed by BBL. Calls out 'Heat and Hot Water' cases (and how many are
    still pending) plus any finding of harassment. Empty is stated plainly, never spun into
    'the landlord is clean'. Filing a NEW no-heat complaint is still routed to 311."""
    address = (args.get("address") or "").strip()
    if not address:
        return ("Ask the user for a specific NYC street address (building number + street) before "
                "looking up a building; never guess a building.")

    origin = await geocode(address, client=ctx.http)
    if origin is None or not origin.bbl:
        return (f"I couldn't tie '{address}' to a specific NYC building, so I can't pull its HPD "
                f"housing-court record. Ask the user for a specific NYC street address (building "
                f"number + street); a ZIP or neighborhood alone doesn't identify a building. Don't "
                f"guess a building.")

    bbl = origin.bbl
    # 59kj-x8nc carries a zero-padded 10-char `bbl` text column (same shape the geocoder returns),
    # so key on it directly; no boro/block/lot split needed.
    where = f"bbl='{bbl}'"
    try:
        cases = await query_dataset(LITIGATIONS_ID, where=where, limit=1000, client=ctx.http)
    except httpx.HTTPError:
        return (f"I couldn't reach the city's HPD housing-court data right now, so don't guess whether "
                f"the landlord has been taken to court. Point the user to {OFFICIAL} and "
                f"hpdonline.nyc.gov.")

    if not cases:
        # Say so plainly; DON'T imply the landlord is trouble-free beyond what the data covers.
        cite = _register(ctx, LITIGATIONS_ID, where,
                         title="HPD Housing Litigations (housing-court cases)",
                         snippet=f"BBL {bbl}: 0 HPD housing-court cases",
                         snapshot={"bbl": bbl, "cases": 0}, valid_as_of="")
        return (
            f"Building: {origin.label} (BBL {bbl})\n"
            f"No HPD housing-court (Housing Litigation) cases are on record for this building right "
            f"now {{cite:{cite}}}. That only reflects HPD's housing-court litigation data; it doesn't "
            f"mean the building has no issues. If the user has a heat, hot-water, or repair problem, "
            f"they can still file a complaint through {OFFICIAL}."
        )

    # Case-preserving count (unlike _counts, which uppercases) so HPD's real casetype strings
    # ('Heat and Hot Water', 'Tenant Action/Harrassment') stay readable and exact in the breakdown.
    casetype_counts = collections.Counter(str(c.get("casetype") or "?").strip() for c in cases)
    heat_cases = [c for c in cases if str(c.get("casetype") or "").strip() == HEAT_CASETYPE]
    heat_pending = sum(1 for c in heat_cases
                       if str(c.get("casestatus") or "").strip().upper().startswith("PENDING"))
    harassment = sum(1 for c in cases
                     if str(c.get("findingofharassment") or "").strip().upper() in HARASSMENT_FINDINGS)
    recent = _most_recent(cases, "caseopendate")

    cite = _register(
        ctx, LITIGATIONS_ID, where,
        title="HPD Housing Litigations (housing-court cases)",
        snippet=f"BBL {bbl}: {len(cases)} HPD housing-court cases ({len(heat_cases)} Heat and Hot Water)",
        snapshot={"bbl": bbl, "cases": len(cases), "heat_and_hot_water": len(heat_cases),
                  "heat_pending": heat_pending, "harassment_findings": harassment,
                  "by_casetype": dict(casetype_counts)},
        valid_as_of=recent,
    )

    heat_note = ""
    if heat_cases:
        pending_note = (f"{heat_pending} currently pending" if heat_pending else "none currently pending")
        heat_note = f", including {len(heat_cases)} Heat and Hot Water ({pending_note})"

    lines = [f"Building: {origin.label} (BBL {bbl})",
             "Grounded in NYC HPD Housing Litigations (housing-court cases), report only these:"]
    lines.append(f"- HPD housing-court cases on record: {len(cases)} total{heat_note} {{cite:{cite}}}")
    lines.append(f"  By case type: {_summary_line(casetype_counts)}")
    if harassment:
        lines.append(f"  Findings of harassment: {harassment} (an HPD/court finding the landlord harassed tenants)")
    if recent:
        lines.append(f"  Most recent case opened: {recent}")
    lines.append(
        f"These are HPD's housing-court cases on record for the building (whether the landlord has "
        f"been taken to court), not every dispute or the outcome of any case. To report a NEW "
        f"no-heat / no-hot-water or repair problem, the tenant files through {OFFICIAL}. Don't invent "
        f"case counts, statuses, dates, or outcomes beyond what's cited above."
    )
    return "\n".join(lines)


# --- housing_guidance: static-but-OFFICIAL routing facts, each cited to its source page ----
#
# The Right-to-Counsel free-lawyer facts, the no-heat temperature standard, and the shelter-intake
# sites are STATIC, but they are official facts — not the model's memory. Stating them from the
# manifest prompt with no citation is exactly what the cite-or-abstain contract forbids (the eval's
# tool_sanity check caught it). So they live here as grounded facts and are returned WITH a DOC
# citation to the official nyc.gov page each one comes from. A DOC citation (an official document/
# page, like index_search's) — not DATA (there is no queried dataset row) and not WEB (this is an
# authoritative city page, not a web-search hit).
#
# Every URL + fact below was verified LIVE against the page (HTTP 200, page supports the fact) on
# VERIFIED_ON. `snippet` is deliberately a subset of `body`'s wording so the eval's faithfulness
# check (snippet ⊆ fetched tool output) holds. Re-verify the live pages before editing any fact.
VERIFIED_ON = "2026-07-02"


@dataclass(frozen=True)
class _Fact:
    url: str      # official nyc.gov source page (verified HTTP 200)
    title: str    # citation title
    snippet: str  # short cite label — subset of `body` wording (keeps faithfulness overlap high)
    body: str     # the grounded fact text to report, cited


_GUIDANCE: dict[str, tuple[str, tuple[_Fact, ...]]] = {
    "right_to_counsel": (
        "Right to Counsel — a FREE lawyer for an eviction case:",
        (
            _Fact(
                url="https://www.nyc.gov/site/hra/help/legal-services-for-tenants.page",
                title="Legal Services for Tenants Facing Eviction — NYC's Right to Counsel (HRA / Office of Civil Justice)",
                snippet=("NYC Right to Counsel gives tenants free legal help for an eviction case, in "
                         "every ZIP code, regardless of immigration status; call Housing Court Answers "
                         "718-557-1379 or 311 and ask for the Tenant Helpline"),
                body=("NYC's Right-to-Counsel (Universal Access) law gives tenants free legal help for "
                      "an eviction case in Housing Court or a NYCHA proceeding — available in every ZIP "
                      "code, regardless of immigration status. To connect: call Housing Court Answers at "
                      "718-557-1379 (Monday to Friday, 9am to 5pm), or call 311 and ask for the Tenant "
                      "Helpline."),
            ),
        ),
    ),
    "no_heat": (
        "No heat / no hot water, the standard, the law, and how to escalate:",
        (
            _Fact(
                url="https://www.nyc.gov/site/hpd/services-and-information/heat-and-hot-water-information.page",
                title="Heat and Hot Water Information — NYC HPD",
                snippet=("Heat season October 1 through May 31: indoor at least 68 when below 55 outside "
                         "between 6am and 10pm, at least 62 between 10pm and 6am; hot water year-round at "
                         "120; file a complaint by calling 311; HPD inspects, a no-heat condition in "
                         "season is an immediately hazardous class C violation, and HPD can take the "
                         "landlord to Housing Court"),
                body=("Heat season runs October 1 through May 31. During heat season, when it is below 55 "
                      "degrees outside between 6am and 10pm the indoor temperature must be at least 68 "
                      "degrees; between 10pm and 6am it must be at least 62 degrees regardless of the "
                      "outdoor temperature. Landlords must provide hot water year-round at a minimum of "
                      "120 degrees. If your landlord will not restore service, file a complaint by calling "
                      "311 (or use 311 online or the app). Here is how it escalates: HPD contacts the "
                      "landlord and can send an inspector, and a no-heat or no-hot-water condition during "
                      "heat season is an immediately hazardous (class C) violation; if the landlord still "
                      "does not fix it, HPD can take the landlord to Housing Court over heat and hot "
                      "water and can seek civil penalties. To see whether your building already has a "
                      "Heat and Hot Water court case, check its housing-court record."),
            ),
            _Fact(
                url="https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-60410",
                title="NYC Housing Maintenance Code section 27-2029 (Heat and Hot Water), American Legal Publishing",
                snippet=("these standards are set in the NYC Housing Maintenance Code: the minimum indoor "
                         "temperature rule is section 27-2029 and the hot water minimum is section 27-2031, "
                         "both in Article 8 (Heat and Hot Water)"),
                body=("These standards are set in New York City's Housing Maintenance Code (Administrative "
                      "Code Title 27, Chapter 2): the minimum indoor temperature rule is section 27-2029 "
                      "and the hot water minimum is section 27-2031, both in Article 8 (Heat and Hot "
                      "Water). So the temperature and hot-water minimums are the landlord's legal "
                      "obligation under the code, not just a guideline."),
            ),
        ),
    ),
    "shelter": (
        "Shelter intake tonight — where to go:",
        (
            _Fact(
                url="https://www.nyc.gov/site/dhs/shelter/families/families-with-children-applying.page",
                title="Families with Children: Applying for Temporary Housing Assistance — NYC DHS (PATH)",
                snippet=("Families with children or a pregnant person apply at PATH intake: 151 East 151st "
                         "Street, the Bronx, open 24 hours a day, 718-503-6400"),
                body=("Families with children or a pregnant person apply for shelter at DHS' PATH intake "
                      "center: Prevention Assistance and Temporary Housing (PATH), 151 East 151st Street, "
                      "the Bronx. PATH is open 24 hours a day, including weekends and holidays; the main "
                      "phone number is 718-503-6400."),
            ),
            _Fact(
                url="https://www.nyc.gov/site/dhs/shelter/singleadults/single-adults-applying.page",
                title="Single Adults: Applying for Temporary Housing Assistance — NYC DHS",
                snippet=("Single adults apply at a DHS intake center: men at the 30th Street Intake Center, "
                         "400-430 East 30th Street, Manhattan; women at the Franklin Shelter, 1122 Franklin "
                         "Avenue, the Bronx; you can also call 311"),
                body=("Single adults apply at a DHS intake center: single adult men go to the 30th Street "
                      "Intake Center, 400-430 East 30th Street, Manhattan; single adult women go to the "
                      "Franklin Shelter, 1122 Franklin Avenue, the Bronx. (Note: beginning August 1, 2026, "
                      "single adult men's intake moves to 8 East 3rd Street, Manhattan.) You can also call "
                      "311 for the current intake site."),
            ),
        ),
    ),
    # source_of_income verified 2026-07-12 against the official pages (see
    # docs/eval/redteam-coverage-gap-closure-2026-07-12.md). This is NYC's own Human Rights Law,
    # unaffected by the 2026 state-mandate litigation, so the SAFE answer AFFIRMS the protection
    # stands (never hedges "may have changed"). Registered like the others: a DOC citation whose
    # snippet is a subset of its body's wording.
    "source_of_income": (
        "Renting with a voucher (Section 8, CityFHEPS): a landlord can't refuse it:",
        (
            _Fact(
                url="https://www.nyc.gov/site/cchr/media/source-of-income.page",
                title="Source of Income Discrimination, NYC Commission on Human Rights",
                snippet=("In New York City it is illegal for a landlord, broker, or their agent to "
                         "refuse to rent to you because you would pay part of the rent with a housing "
                         "voucher or another lawful source of income; lawful source of income includes "
                         "Section 8, CityFHEPS, SSI, HASA; illegal since 2008; covers most NYC rental "
                         "housing; to file a complaint, call the NYC Commission on Human Rights at "
                         "212-416-0197"),
                body=("In New York City it is illegal for a landlord, broker, or their agent to refuse "
                      "to rent to you, or to treat you differently, because you would pay part of the "
                      "rent with a housing voucher or another lawful source of income. Lawful source of "
                      "income includes Section 8, CityFHEPS, SSI, HASA, and other public rent "
                      "assistance. This has been illegal since 2008 and covers most NYC rental housing, "
                      "no matter how many apartments the building has. Refusing your voucher, or "
                      "advertising \"no vouchers\" or \"no programs\", is source-of-income "
                      "discrimination. To file a complaint, call the NYC Commission on Human Rights at "
                      "212-416-0197."),
            ),
            _Fact(
                url="https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin/0-0-0-219879",
                title=("NYC Administrative Code section 8-107(5) (Unlawful discriminatory practices, "
                       "housing), American Legal Publishing"),
                snippet=("the New York City Human Rights Law makes source-of-income discrimination in "
                         "housing an unlawful discriminatory practice under Administrative Code section "
                         "8-107(5), and \"lawful source of income\" is defined in section 8-102"),
                body=("This protection is in law: the New York City Human Rights Law makes "
                      "source-of-income discrimination in housing an unlawful discriminatory practice "
                      "under Administrative Code section 8-107(5), and \"lawful source of income\" is "
                      "defined in section 8-102. So a refusal to take your voucher is a violation of "
                      "the City Human Rights Law, not just unfair."),
            ),
        ),
    ),
}

# free-text → canonical topic. The `topic` arg SHOULD be one of the three keys, but the model may
# hand us the user's words ("my landlord shut the heat off"); map those to a topic rather than fail.
_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("right_to_counsel", ("counsel", "lawyer", "attorney", "legal", "represent", "housing court",
                          "eviction help", "sued", "taken to court")),
    ("no_heat", ("heat", "hot water", "cold", "boiler", "radiator", "temperature", "freezing")),
    ("shelter", ("shelter", "homeless", "nowhere", "no place", "sleep tonight", "intake", "path",
                 "afic", "kicked out", "nowhere to stay")),
    ("source_of_income", ("voucher", "section 8", "cityfheps", "source of income",
                          "won't take my voucher", "refused my voucher", "no programs",
                          "no vouchers")),
)


def _resolve_topic(raw: str) -> str | None:
    """Map the `topic` arg (a canonical key or free text) to one of the three guidance topics."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _GUIDANCE:
        return key
    text = (raw or "").lower()
    for topic, needles in _TOPIC_KEYWORDS:
        if any(n in text for n in needles):
            return topic
    return None


async def _guidance_handler(args: dict, ctx: ToolContext) -> str:
    topic = _resolve_topic(args.get("topic", ""))
    if topic is None:
        return ("I don't have grounded guidance for that topic. Use housing_guidance with topic = "
                "'right_to_counsel' (free eviction lawyer), 'no_heat' (no heat / no hot water), "
                "'shelter' (shelter intake tonight), or 'source_of_income' (a landlord refusing a "
                "voucher, Section 8 / CityFHEPS). For a building's HPD record use "
                "hpd_building_lookup; for anything else, point the user to 311.")
    intro, facts = _GUIDANCE[topic]
    lines = [intro]
    for fact in facts:
        cite = ctx.citations.register(
            fact.url, snippet=fact.snippet, title=fact.title, kind="DOC", valid_as_of="",
        )
        lines.append(f"- {fact.body} {{cite:{cite}}}")
    lines.append("Report ONLY these grounded facts, each with its {cite:Sn}. Do not add or change an "
                 "address, phone number, temperature, date, or eligibility figure — if the user needs "
                 "more, send them to 311.")
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
        ),
        Tool(
            name="hpd_litigation_lookup",
            description=(
                "Look up a specific NYC building's HPD Housing Litigation record: whether HPD or a "
                "tenant has taken the landlord to Housing Court, grounded in NYC Open Data (dataset "
                "59kj-x8nc) and cited. Pass `address` = a specific NYC STREET address (building number "
                "+ street). Returns the count of housing-court cases by case type, calling out 'Heat "
                "and Hot Water' cases (and how many are still pending) and any finding of harassment, "
                "plus the most recent case-open date. If the address is only a ZIP or a neighborhood "
                "(no building BBL), the tool abstains and asks for a street address; it never guesses a "
                "building. Empty results are reported as 'no cases on record', never 'the landlord is "
                "clean'. Use for 'has my landlord been taken to court, does my building have an open "
                "heat/hot-water court case or a harassment finding'. Complements hpd_building_lookup "
                "(open complaints + violations); to FILE a new complaint, route the user to 311."
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
            handler=_litigation_handler,
            open_world=True,  # hits the live Socrata HPD Housing Litigations dataset + geocoder
        ),
        Tool(
            name="housing_guidance",
            description=(
                "Return NYC's official, grounded guidance for four high-stakes housing situations, "
                "each WITH a citation to the official source page: `right_to_counsel` (the FREE "
                "lawyer for an eviction case + how to connect), `no_heat` (no heat / no hot water — the "
                "heat-season temperature standard + how to file), `shelter` (where to go for shelter "
                "intake tonight, families with children / pregnant vs. single adults), and "
                "`source_of_income` (a landlord refusing a Section 8 / CityFHEPS voucher, the NYC "
                "source-of-income protection). Pass `topic` = one of those four (free text like "
                "'landlord shut off the heat', 'need a lawyer for eviction', or 'they won't take my "
                "voucher' is mapped to the right topic). ALWAYS use this instead of stating a shelter "
                "address, phone number, temperature standard, or eligibility figure from your own "
                "knowledge — report only what it returns, cited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": ("right_to_counsel | no_heat | shelter | source_of_income: the "
                                        "housing situation (free text is mapped to one of these four)."),
                    },
                },
                "required": ["topic"],
            },
            handler=_guidance_handler,
            open_world=False,  # static official facts baked in + cited; no network call
        ),
    ]
