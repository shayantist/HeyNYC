"""clinics module tool: `find_clinic`, the nearest NYC safety-net clinics that will see you
regardless of insurance or immigration status.

Two source CLASSES, merged and ranked by distance:

  - FQHC (live): HRSA's Primary Health Care service-delivery sites (a public, tokenless ArcGIS
    MapServer layer), filtered to ACTIVE sites in the five NYC counties (~459). Federally Qualified
    Health Centers are required by the HRSA Health Center Program to serve everyone on a sliding fee
    scale regardless of ability to pay.
  - NYC_CARE (bundled seed): the 11 NYC Health + Hospitals acute-care hospitals + the Gotham Health
    community health centers, transcribed from nychealthandhospitals.org/locations and geocoded at
    BUILD time (see build_seed.py). NYC Care guarantees low/no-cost care at H+H and doesn't ask about
    immigration status.

ANTI-HALLUCINATION CORE: the eligibility / cost / immigration-safety framing NEVER comes from a
per-row field (per-row cost data is sparse and malformed) and NEVER from the model. It comes ONLY
from the CLASS -> ProgramGuarantee map below, whose wording is grounded to and cited from the
program's official page (hrsa.gov for FQHC, access.nyc.gov for NYC Care). Each returned site carries
a DATA citation (the facility source) AND its class's DOC citation (the program page).

If geocoding fails or nothing is near, the tool abstains and routes to 311 / 646-NYC-CARE, it never
guesses a clinic.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    _clarify_message,
    _resolution_note,
    format_distance,
    geocode,
    haversine_m,
    maps_link,
    miles,
)

# --- FQHC spine (live HRSA ArcGIS MapServer layer; behaves like a FeatureServer for /query) ---
# Recon-verified 2026-07-04: `f=geojson` returns point features; the generic arcgis client accepts
# this MapServer URL unchanged (it does string ops on the URL, no "FeatureServer" hardcoding).
HRSA_URL = (
    "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
    "PrimaryHealthCareFacilities_FS/MapServer/0"
)
# Active FQHC service-delivery sites in the five NYC counties. Verified count = 459 (2026-07-04):
# Bronx 148, Kings 157, New York 92, Queens 46, Richmond 16.
FQHC_WHERE = ("HCC_STATUS_DESC='Active' AND SITE_STATE_ABBR='NY' "
              "AND COUNTY_NM IN ('Bronx','Kings','New York','Queens','Richmond')")

SEED_PATH = Path(__file__).resolve().parent / "data" / "nyc_care_sites.tsv"
NYC_CARE_SOURCE = "https://www.nychealthandhospitals.org/locations/"
OFFICIAL = "call 311, or 646-NYC-CARE (646-692-2273) for NYC Care enrollment"
# HRSA county names -> common NYC borough names (Kings=Brooklyn, New York=Manhattan, Richmond=SI).
_COUNTY_TO_BOROUGH = {
    "Bronx": "Bronx", "Kings": "Brooklyn", "New York": "Manhattan",
    "Queens": "Queens", "Richmond": "Staten Island",
}

CLASS_FQHC = "FQHC"
CLASS_NYC_CARE = "NYC_CARE"

# The anti-hallucination bar: reviewed + cited on VERIFIED_ON against the official program page.
# `snippet` is a subset of `body`'s wording (keeps the eval faithfulness overlap high). Re-verify
# the live pages before editing any fact.
VERIFIED_ON = "2026-07-04"


@dataclass(frozen=True)
class ProgramGuarantee:
    """A CLASS's grounded eligibility framing + the citations that back it.

    doc_url/doc_title/snippet/body -> the DOC citation to the official PROGRAM page (the eligibility
    guarantee). data_title -> the title for each facility's DATA citation (the facility source).
    """
    label: str
    lead: str          # a warm one-line reassurance shown once per class present in the results
    doc_url: str       # official program page (verified live)
    doc_title: str
    snippet: str       # short cite label, subset of `body`
    body: str          # the grounded eligibility / cost / immigration-safety sentence(s), cited
    data_title: str    # citation title for the facility (DATA) source


CLASS_GUARANTEE: dict[str, ProgramGuarantee] = {
    CLASS_FQHC: ProgramGuarantee(
        label="Community Health Center (FQHC)",
        lead="These are federally funded health centers that see everyone, insured or not.",
        doc_url="https://www.hrsa.gov/get-health-care",
        doc_title="Get Health Care, HRSA Health Center Program",
        snippet=("HRSA-funded health centers see all patients regardless of ability to pay and "
                 "charge on a sliding fee scale based on your income and family size"),
        body=("Federally Qualified Health Centers (community health centers) are funded by HRSA's "
              "Health Center Program to provide primary care in underserved communities. They see "
              "all patients regardless of ability to pay, whether or not you have insurance, and "
              "charge on a sliding fee scale (discounts based on your income and family size), so "
              "cost is not a barrier to care."),
        data_title="HRSA Primary Health Care service-delivery sites",
    ),
    CLASS_NYC_CARE: ProgramGuarantee(
        label="NYC Health + Hospitals (NYC Care)",
        lead="These city hospitals and clinics serve everyone and don't ask about immigration status.",
        doc_url="https://access.nyc.gov/programs/nyc-care/",
        doc_title="NYC Care, ACCESS NYC",
        snippet=("NYC Care gives low- or no-cost care at NYC Health + Hospitals, sliding-scale fees "
                 "starting at $0, and doesn't ask about immigration status; enroll at 646-NYC-CARE "
                 "(646-692-2273)"),
        body=("NYC Care is a health-access program that gives you your own doctor and services at "
              "NYC Health + Hospitals locations citywide, with sliding-scale fees starting at $0 and "
              "no membership fees, monthly fees, or premiums. NYC Care doesn't ask about immigration "
              "status, you can seek care regardless of immigration status or ability to pay. To "
              "enroll, call 646-NYC-CARE (646-692-2273)."),
        data_title="NYC Health + Hospitals locations (NYC Care sites)",
    ),
}


def _clean(value) -> str:
    """None / literal 'NULL' / 'N/A' / blanks -> '' (HRSA uses these as empty markers)."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() in ("NULL", "N/A", "NA") else text


@dataclass
class Clinic:
    name: str
    lat: float
    lon: float
    address: str
    borough: str
    phone: str
    url: str
    klass: str
    record_id: str
    valid_as_of: str
    raw: dict = field(default_factory=dict)


# --- FQHC (live ArcGIS record) ---------------------------------------------

def _fqhc_address(record: dict) -> str:
    """Assemble 'street, City ZIP5' from the HRSA fields (any may be blank)."""
    street = _clean(record.get("SITE_ADDRESS"))
    city = _clean(record.get("SITE_CITY"))
    zip5 = _clean(record.get("SITE_ZIP_CD"))[:5]
    tail = " ".join(p for p in (city, zip5) if p)
    return ", ".join(p for p in (street, tail) if p)


def _fqhc_from_record(record: dict) -> Clinic | None:
    """Map a raw HRSA feature record to an FQHC Clinic; drop records without usable coordinates."""
    try:
        lat = float(record["lat"])
        lon = float(record["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    county = _clean(record.get("COUNTY_NM"))
    return Clinic(
        name=_clean(record.get("SITE_NM")) or "Community health center",
        lat=lat,
        lon=lon,
        address=_fqhc_address(record),
        borough=_COUNTY_TO_BOROUGH.get(county, county),
        phone=_clean(record.get("SITE_PHONE_NUM")),
        url=_clean(record.get("SITE_URL")),
        klass=CLASS_FQHC,
        record_id=_clean(record.get("OBJECTID")),
        valid_as_of=VERIFIED_ON,
        raw=record,
    )


# --- NYC_CARE (bundled, build-time-geocoded seed) --------------------------

def _load_nyc_care_seed(path: Path = SEED_PATH) -> list[Clinic]:
    """Load the bundled NYC Care / H+H seed (build-time geocoded). Rows without coords are dropped.

    A missing/unreadable seed degrades to [] (the tool still serves live FQHCs), never crashes.
    """
    clinics: list[Clinic] = []
    try:
        with path.open(encoding="utf-8") as fh:
            rows = csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t")
            for row in rows:
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                street = _clean(row.get("street"))
                zip5 = _clean(row.get("zip"))[:5]
                borough = _clean(row.get("borough"))
                address = ", ".join(p for p in (street, " ".join(x for x in (borough, zip5) if x)) if p)
                clinics.append(Clinic(
                    name=_clean(row.get("name")) or "NYC Health + Hospitals site",
                    lat=lat,
                    lon=lon,
                    address=address,
                    borough=borough,
                    phone=_clean(row.get("phone")),
                    url=_clean(row.get("url")),
                    klass=CLASS_NYC_CARE,
                    record_id=_clean(row.get("name")),
                    valid_as_of=VERIFIED_ON,
                    raw=dict(row),
                ))
    except OSError:
        return []
    return clinics


# --- citations -------------------------------------------------------------

def _facility_citation(ctx: ToolContext, clinic: Clinic, *,
                       origin_lat: float, origin_lon: float, dist_mi: float) -> str:
    """A DATA citation for the facility source (re-fetchable + provenance + distance derivation).

    FQHC -> the single-feature HRSA ArcGIS permalink (row-addressed by OBJECTID). NYC_CARE -> the
    H+H locations page the seed row was transcribed from (the seed row snapshot is the provenance).
    """
    guarantee = CLASS_GUARANTEE[clinic.klass]
    if clinic.klass == CLASS_FQHC:
        url = (feature_query_url(HRSA_URL, clinic.record_id, id_field="OBJECTID")
               if clinic.record_id else HRSA_URL)
    else:
        url = NYC_CARE_SOURCE
    provenance = data_provenance(
        clinic.raw,
        record_id=clinic.record_id,
        field_pointer="/",
        derivation={"origin": [origin_lat, origin_lon], "point": [clinic.lat, clinic.lon],
                    "distance_mi": dist_mi},
    )
    return ctx.citations.register(
        url,
        snippet=f"{clinic.name}, {clinic.address or clinic.borough or 'NYC'}",
        title=guarantee.data_title,
        kind="DATA",
        valid_as_of=clinic.valid_as_of,
        provenance=provenance,
    )


def _program_citation(ctx: ToolContext, klass: str) -> str:
    """A DOC citation for the CLASS's official program page, the grounded eligibility guarantee.

    Deduped by the registry on (kind, url, snippet), so many sites of one class share one program id.
    """
    guarantee = CLASS_GUARANTEE[klass]
    return ctx.citations.register(
        guarantee.doc_url,
        snippet=guarantee.snippet,
        title=guarantee.doc_title,
        kind="DOC",
        valid_as_of=VERIFIED_ON,
    )


# --- the tool --------------------------------------------------------------

def _clinic_block(clinic: Clinic, cite: str, distance: str) -> str:
    guarantee = CLASS_GUARANTEE[clinic.klass]
    where = clinic.address or clinic.borough or "NYC"
    parts = [f"- {clinic.name} [{guarantee.label}] ({where}), "
             f"{distance} {{cite:{cite}}}"]
    if clinic.phone:
        parts.append(f"  Phone: {clinic.phone}")
    if clinic.url:
        parts.append(f"  Website: {clinic.url}")
    parts.append(f"  Directions: {maps_link(clinic.lat, clinic.lon)}")
    parts.append(f"  As of: {clinic.valid_as_of}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    if not near:
        return ("Ask the user where they are (an NYC address, neighborhood, or ZIP) before searching "
                ", never guess a clinic location.")

    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return (f"I couldn't locate '{near}' in NYC, so I can't find a nearby clinic. Ask the user "
                f"for a specific NYC address, neighborhood, or ZIP, don't guess a clinic. If they "
                f"need care now, they can {OFFICIAL}.")
    if origin.low_confidence:
        return _clarify_message(near)

    # FQHC spine is live; on any HRSA error we DEGRADE to the bundled NYC Care seed rather than
    # abstain (the safety-net answer still stands). Only a truly empty merge abstains.
    fqhcs: list[Clinic] = []
    degraded = False
    try:
        records = await query_feature_service(HRSA_URL, where=FQHC_WHERE, client=ctx.http)
        fqhcs = [c for c in (_fqhc_from_record(r) for r in records) if c is not None]
    except httpx.HTTPError:
        degraded = True

    clinics = fqhcs + _load_nyc_care_seed()
    if not clinics:
        return (f"I couldn't pull any NYC safety-net clinics right now, don't invent one. Point the "
                f"user to {OFFICIAL}, or findahealthcenter.hrsa.gov.")

    k = int(args.get("k") or 5)
    ordered = sorted(clinics, key=lambda c: haversine_m(origin.lat, origin.lon, c.lat, c.lon))
    ranked: list[Clinic] = []
    seen: set[tuple] = set()
    for clinic in ordered:
        key = (clinic.name.strip().lower(), round(clinic.lat, 5), round(clinic.lon, 5))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(clinic)
        if len(ranked) >= k:
            break

    lines = [
        f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})",
        _resolution_note(near, origin),
    ]
    if degraded:
        lines.append("(HRSA's live health-center data was unreachable, showing NYC Health + "
                     "Hospitals / NYC Care sites only. Suggest the user also try findahealthcenter.hrsa.gov.)")
    lines.append("Nearby clinics that will see you regardless of insurance, report only these, cite "
                 "each. Lead with the reassurance that these places serve everyone:")
    classes_present: list[str] = []
    for clinic in ranked:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, clinic.lat, clinic.lon))
        cite = _facility_citation(ctx, clinic, origin_lat=origin.lat, origin_lon=origin.lon,
                                  dist_mi=dist_mi)
        lines.append(_clinic_block(clinic, cite, format_distance(near, origin, dist_mi)))
        if clinic.klass not in classes_present:
            classes_present.append(clinic.klass)

    # The eligibility / immigration-safety framing, ONE grounded, cited block per class present.
    # This is the ONLY source of any cost/eligibility/immigration claim (never a per-row field).
    for klass in classes_present:
        guarantee = CLASS_GUARANTEE[klass]
        prog_cite = _program_citation(ctx, klass)
        lines.append(f"What {guarantee.label} means for you: {guarantee.body} {{cite:{prog_cite}}}")

    lines.append("Call ahead to confirm hours and the services you need, this is not medical advice, "
                 "and for a medical emergency call 911. Report only the sites and the grounded "
                 "eligibility text above; never state a cost, eligibility, or immigration fact that "
                 "isn't in this result.")
    return "\n".join(lines)


# --- health_coverage_guidance: static-but-OFFICIAL coverage facts, each cited to its source page ---
#
# `find_clinic` answers "where can I get seen"; this answers "what coverage can I get, and is it
# safe" for an uninsured or undocumented New Yorker. The facts are STATIC but official (a program's
# guarantee, a state coverage rule), so, like housing_guidance, they live here as grounded _Fact
# records returned WITH a DOC citation to the official page each one comes from, never stated from
# the model's memory. Verified 2026-07-12 against the linked pages (see
# docs/internal/eval/redteam-coverage-gap-closure-2026-07-12.md); `snippet` is a subset of `body`'s wording
# so the eval's faithfulness check (snippet ⊆ tool output) holds. Where the question crosses into
# immigration-law consequences (public charge), the tool ROUTES to ActionNYC rather than assert a
# volatile immigration-law conclusion.
COVERAGE_VERIFIED_ON = "2026-07-25"

COVERAGE_INTRO = "Health coverage you can get in NYC regardless of immigration status:"

# Appended once at the end of every coverage answer: keeps the public-charge (immigration-law)
# question out of the model's mouth and routes it to free, confidential, trusted legal help.
COVERAGE_CLOSING = (
    "Whether using a public benefit could ever affect an immigration case (the \"public charge\" "
    "question) is a legal question with rules that change; get free, confidential, trusted advice "
    "through ActionNYC (call 311 and ask for ActionNYC) before deciding, and don't act on rumors."
)


@dataclass(frozen=True)
class _Fact:
    """A static-but-official coverage fact + the DOC citation that backs it. `snippet` is a subset
    of `body`'s wording (keeps the faithfulness overlap high)."""
    url: str      # official program / coverage page (verified)
    title: str    # citation title
    snippet: str  # short cite label, a subset of `body`
    body: str     # the grounded coverage fact to report, cited


_COVERAGE: dict[str, _Fact] = {
    "emergency_care": _Fact(
        url="https://www.cms.gov/priorities/your-patient-rights/emergency-room-rights",
        title="Emergency room rights under EMTALA, Centers for Medicare & Medicaid Services",
        snippet=("A hospital emergency department cannot deny an emergency screening or stabilizing "
                 "treatment because of insurance status or ability to pay; this does not mean the care "
                 "is free, and Emergency Medicaid eligibility is a separate question"),
        body=("A hospital emergency department cannot deny an emergency screening or stabilizing "
              "treatment because of your insurance status or ability to pay. This does not mean the "
              "care is free, and Emergency Medicaid eligibility is a separate question. If this may "
              "be an emergency, go to an emergency department or call 911."),
    ),
    "nyc_care": _Fact(
        url="https://access.nyc.gov/programs/nyc-care/",
        title="NYC Care, ACCESS NYC",
        snippet=("NYC Care gives low- or no-cost care at NYC Health + Hospitals, sliding-scale fees "
                 "starting at $0, and doesn't ask about immigration status; enroll at 646-NYC-CARE "
                 "(646-692-2273)"),
        body=("NYC Care is a health-access program that gives you your own doctor and services at "
              "NYC Health + Hospitals locations citywide, with sliding-scale fees starting at $0 and "
              "no membership fees, monthly fees, or premiums. NYC Care doesn't ask about immigration "
              "status; you can seek care regardless of immigration status or ability to pay. To "
              "enroll, call 646-NYC-CARE (646-692-2273)."),
    ),
    "emergency_medicaid": _Fact(
        url="https://www.health.ny.gov/health_care/medicaid/emergency_medical_condition_faq.htm",
        title=("Medicaid Emergency Services Only, Treatment of an Emergency Medical Condition, NY "
               "State Department of Health"),
        snippet=("Emergency Medicaid helps eligible New Yorkers, including undocumented immigrants, "
                 "pay for care for a medical emergency regardless of immigration status, if they meet "
                 "the other Medicaid rules for income, identity, and New York State residence; it "
                 "covers emergency labor and delivery and kidney dialysis; you can apply up to three "
                 "months after the emergency care"),
        body=("Emergency Medicaid (Medicaid for the treatment of an emergency medical condition) "
              "helps eligible New Yorkers, including undocumented immigrants, pay for care for a "
              "medical emergency, regardless of immigration status, as long as they meet the other "
              "Medicaid rules for income, identity, and New York State residence. It covers the "
              "treatment of a sudden, serious medical condition, including emergency labor and "
              "delivery and kidney dialysis. You can apply up to three months after the emergency "
              "care. In NYC, apply through HRA."),
    ),
    # Volatile legal-review item, verified 2026-07-13. Re-check the MOIA page before trusting it.
    "public_charge": _Fact(
        url="https://www.nyc.gov/site/immigrants/legal-resources/public-charge-rule.page",
        title="Public Charge Rule, NYC Mayor's Office of Immigrant Affairs (MOIA)",
        snippet=("Before September 18, 2026, the 2022 public charge rule remains in effect: SNAP, "
                 "WIC, housing help, and Medicaid other than long-term institutional care do not count "
                 "against you. Because your own case can be specific, confirm with free, confidential "
                 "advice through ActionNYC or the MOIA immigration hotline at 800-354-0365."),
        body=("Before September 18, 2026, the 2022 public charge rule remains in effect. Under that "
              "rule, SNAP, WIC, housing help, and Medicaid other than long-term institutional care do "
              "not count against you; immigration officials consider cash assistance for income "
              "support and long-term government-funded institutional care. Public charge does not "
              "apply to every immigration situation. "
              "Because your own case can be specific and these rules can change, confirm with free, "
              "confidential, trusted advice through ActionNYC (call 311 and ask for ActionNYC) or the "
              "MOIA immigration hotline at 800-354-0365 before you decide."),
    ),
}

PUBLIC_CHARGE_FINAL_RULE = _Fact(
    url=("https://www.federalregister.gov/documents/2026/07/20/2026-14539/"
         "public-charge-ground-of-inadmissibility"),
    title="Public Charge Ground of Inadmissibility, DHS final rule (Federal Register)",
    snippet=("DHS published a final rule effective September 18, 2026. Benefits received before that "
             "date remain governed by the 2022 rule; for covered applications on or after that date, "
             "officers may consider means-tested public benefits in an individualized review."),
    body=("DHS published a final rule effective September 18, 2026. Benefits received before that "
          "date remain governed by the 2022 rule. For applications for admission made on or after "
          "September 18, or adjustment-of-status applications submitted on or after that date, "
          "officers may consider receipt of means-tested public benefits as part of an individualized, "
          "case-specific review. DHS said additional implementation guidance would follow."),
)

# free-text → canonical coverage topic (the model may hand us the user's words instead of a key).
_COVERAGE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("emergency_care", ("tourist visa", "tourist", "visitor", "emergency medical care",
                        "denied emergency care", "emergency treatment")),
    ("emergency_medicaid", ("emergency medicaid", "emergency room", "er bill", "hospital bill",
                            "labor and delivery", "giving birth", "delivery", "dialysis",
                            "emergency medical", "ambulance bill")),
    ("public_charge", ("public charge", "green card", "green-card", "hurt my green card",
                       "affect my green card", "affect my immigration", "hurt my immigration",
                       "immigration case", "will using benefits", "public benefit immigration")),
    ("nyc_care", ("nyc care", "nyccare", "646-nyc-care", "own doctor", "primary care", "coverage",
                  "health insurance", "get insurance", "sign up for insurance", "no insurance")),
)


def _resolve_coverage_topic(raw: str) -> str | None:
    """Map the `topic` arg (a canonical key or free text) to one of the coverage topics."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _COVERAGE:
        return key
    text = (raw or "").lower()
    for topic, needles in _COVERAGE_KEYWORDS:
        if any(n in text for n in needles):
            return topic
    return None


async def _coverage_handler(args: dict, ctx: ToolContext) -> str:
    topic = _resolve_coverage_topic(args.get("topic", ""))
    if topic is None:
        return ("I don't have grounded coverage guidance for that. Use health_coverage_guidance with "
                "topic = 'emergency_care' (ER screening and stabilizing care), 'nyc_care' (low/no-cost care at NYC Health + Hospitals, no immigration "
                "questions) or 'emergency_medicaid' (coverage for a medical emergency regardless of "
                "immigration status). To find a specific clinic use find_clinic; for anything else, "
                "point the user to 311 or 646-NYC-CARE (646-692-2273).")
    facts = (_COVERAGE[topic],)
    if topic == "public_charge":
        facts += (PUBLIC_CHARGE_FINAL_RULE,)
    lines = [COVERAGE_INTRO]
    for fact in facts:
        cite = ctx.citations.register(fact.url, snippet=fact.snippet, title=fact.title, kind="DOC",
                                      valid_as_of=COVERAGE_VERIFIED_ON)
        lines.append(f"- {fact.body} {{cite:{cite}}}")
    lines.extend([
        COVERAGE_CLOSING,
        "Report ONLY these grounded facts with their {cite:Sn} and the ActionNYC routing line above. Do "
        "not add or change a phone number, a dollar figure, or an eligibility rule, and do not state "
        "a public-charge conclusion of your own; if the user needs more, that's 311 / 646-NYC-CARE / "
        "ActionNYC.",
    ])
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_clinic",
            description=(
                "Find the nearest NYC safety-net clinics that will see someone regardless of "
                "insurance or immigration status, grounded + cited. Merges two sources: live HRSA "
                "Federally Qualified Health Centers (community health centers, sliding fee scale) and "
                "NYC Health + Hospitals / NYC Care sites (low/no-cost, doesn't ask immigration "
                "status). Pass `near` = the user's NYC address, neighborhood, or ZIP; optional `k` "
                "(default 5). Returns each site's name, address, borough, phone, distance, and CLASS, "
                "plus a grounded eligibility guarantee cited to the program's official page. Use for "
                "'doctor without insurance', 'free clinic', 'I'm undocumented and sick'. NEVER guess a "
                "clinic: if geocoding fails or none are near, it abstains and routes to 311 / "
                "646-NYC-CARE. The eligibility/immigration-safety text comes only from the program "
                "citation, never invented per-site."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string",
                             "description": "The NYC address, neighborhood, or ZIP to search near."},
                    "k": {"type": "integer",
                          "description": "How many clinics to return (default 5).", "default": 5},
                },
                "required": ["near"],
            },
            handler=_handler,
            open_world=True,  # hits the live HRSA ArcGIS service + geocoder (NYC Care seed is bundled)
        ),
        Tool(
            name="health_coverage_guidance",
            description=(
                "Answer WHAT health coverage an uninsured or undocumented New Yorker can get, and "
                "whether it's SAFE to use, grounded + cited to the official page. Topics: `nyc_care` "
                "(low/no-cost care at NYC Health + Hospitals, sliding-scale fees from $0, doesn't ask "
                "immigration status, enroll at 646-NYC-CARE) and `emergency_medicaid` (Medicaid for a "
                "medical emergency, including emergency labor and delivery, regardless of immigration "
                "status), `emergency_care` (ER screening and stabilizing care regardless of ability "
                "to pay), and `public_charge` (MOIA public-charge guidance). Pass `topic` = one of "
                "those (free text like 'undocumented and pregnant, how "
                "do I pay for the delivery' is mapped to the right topic). find_clinic answers WHERE "
                "to go; this answers WHAT coverage / IS IT SAFE. It appends an ActionNYC routing line "
                "for public-charge questions; never state a coverage rule or a public-charge conclusion "
                "from your own knowledge; report only what it returns, cited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": ("emergency_care | nyc_care | emergency_medicaid | public_charge: the coverage "
                                        "situation (free text is mapped to one of these three)."),
                    },
                },
                "required": ["topic"],
            },
            handler=_coverage_handler,
            open_world=False,  # static official facts baked in + cited; no network call
        ),
    ]
