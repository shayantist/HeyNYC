"""childcare module tool: `nearest_child_care`, grounded in the DOHMH regulated child care list.

Data source: the official NYC Open Data dataset "Active NYC Health Code Regulated Child Care
Programs" (`gy3q-4tzp` on data.cityofnewyork.us). This is the same Health Department data that powers
NYC Child Care Connect: every center-based child care program permitted and inspected by DOHMH under
Health Code Article 47 (Group Day Care) and Article 43 (School-based Child Care). We fetch the NYC
rows, rank them by Haversine distance from the user's geocoded location (reused geo machinery), and
return the closest few with: program name, full address, phone, the age range served, the maximum
licensed capacity, the facility type (group vs. school-based child care), and the program type
(infant/toddler vs. preschool). Every program is a row-addressed DATA citation resolving to its
Socrata row permalink.

Honest limitations (enforced in the manifest prompt too):
  - `capacity` is the program's MAXIMUM licensed capacity (set by room square footage, toilets and
    sinks) - NOT the number of open spots. A program may be full. We never present it as availability.
  - The source has NO hours, NO cost/tuition, and NO current-openings field - we never invent them;
    we tell the user to call the program.
  - A DOHMH permit is a health regulation, not an endorsement or a quality rating (the Health
    Department explicitly does not endorse). We never assert quality or safety beyond "permitted."
  - This is center-based care only. It does NOT include NY-State-regulated home/family child care or
    after-school (OCFS), and it does NOT show free 3-K/Pre-K seats or child care vouchers - we point
    to MySchools and ACCESS NYC for those, we never assert a child qualifies from this data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import httpx

from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset, row_url
from heynyc.core.tools.geo import (
    _clarify_message,
    _resolution_note,
    geocode,
    haversine_m,
    miles,
)

# The live backend of NYC Child Care Connect - verified public + tokenless on data.cityofnewyork.us.
CHILDCARE_DATASET = "gy3q-4tzp"
# Only rows with a usable coordinate (the source has a handful with none); we never guess a location.
WHERE_HAS_COORDS = "latitude IS NOT NULL AND longitude IS NOT NULL"
OFFICIAL = ("NYC Child Care Connect at nyc.gov/site/doh/services/child-care.page or call 311 "
            "(for free 3-K/Pre-K, MySchools.nyc; for child care vouchers, ACCESS NYC)")

# Verified against the dataset's own column descriptions (DOHMH): facility_type is GCC (group child
# care, Article 47) or SBCC (school-based child care, Article 43). We only ever surface a label we
# can ground in the source code; an unknown code yields no label rather than an invented one.
_FACILITY_LABELS = {"GCC": "group child care", "SBCC": "school-based child care"}
_PROGRAM_LABELS = {"PRESCHOOL": "preschool", "INFANT TODDLER": "infant/toddler"}


def _clean(value) -> str:
    """None / literal 'NULL' / 'NO DATA' / blanks -> ''. The source uses these as empty sentinels."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() in ("NULL", "NO DATA") else text


def _parse_coords(record: dict) -> tuple[float, float] | None:
    """(lat, lon) from the flat `latitude`/`longitude` fields, or None if absent/unparseable.

    We never guess a coordinate: a missing or malformed pair drops the row from the results."""
    try:
        return float(record["latitude"]), float(record["longitude"])
    except (KeyError, TypeError, ValueError):
        return None


def _address(record: dict) -> str:
    """Assemble the street address from the source's parts (any of which may be blank)."""
    tail = " ".join(p for p in (_clean(record.get("borough")), "NY", _clean(record.get("zipcode"))) if p)
    return ", ".join(p for p in (_clean(record.get("address")), tail) if p)


def _facility_label(record: dict) -> str:
    return _FACILITY_LABELS.get(_clean(record.get("facility_type")).upper(), "")


def _program_label(record: dict) -> str:
    return _PROGRAM_LABELS.get(_clean(record.get("program_type")).upper(), "")


def _age_range(record: dict) -> str:
    """The served age range, or '' for the source's 'NO DATA' sentinel (never shown to the user)."""
    return _clean(record.get("age_range"))


def _capacity_phrase(record: dict) -> str:
    """A max-licensed-capacity phrase, framed so it is never mistaken for open spots. '' if blank."""
    cap = _clean(record.get("capacity"))
    if not cap:
        return ""
    return f"licensed for up to {cap} children (max capacity, not open spots)"


def _valid_as_of(record: dict) -> str:
    """The row's Socrata `:updated_at` change signal, or blank when unavailable."""
    text = _clean(record.get(":updated_at"))
    if text:
        try:
            return datetime.fromisoformat(text[:10]).date().isoformat()
        except ValueError:
            pass
    return ""


# --- record -> site --------------------------------------------------------

@dataclass
class ChildCareSite:
    name: str
    lat: float
    lon: float
    address: str
    phone: str
    age_range: str
    capacity_phrase: str
    facility_label: str
    program_label: str
    administer_medication: str
    row_id: str
    valid_as_of: str
    raw: dict = field(default_factory=dict)


def _to_site(record: dict) -> ChildCareSite | None:
    """Map a raw Socrata record to a ChildCareSite; drop records without usable coordinates."""
    coords = _parse_coords(record)
    if coords is None:
        return None
    lat, lon = coords
    return ChildCareSite(
        name=_clean(record.get("program_name")) or "Child care program",
        lat=lat,
        lon=lon,
        address=_address(record),
        phone=_clean(record.get("phone")),
        age_range=_age_range(record),
        capacity_phrase=_capacity_phrase(record),
        facility_label=_facility_label(record),
        program_label=_program_label(record),
        administer_medication=_clean(record.get("administer_medication")),
        row_id=_clean(record.get(":id")),
        valid_as_of=_valid_as_of(record),
        raw=record,
    )


def directions_link(lat: float, lon: float) -> str:
    """A Google Maps directions deep link to a grounded coordinate (navigation handoff, no citation
    needed - it's a deterministic transform of an already-grounded point)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.5f},{lon:.5f}"


async def _query_childcare(client: httpx.AsyncClient, *, limit: int = 5000) -> list[dict]:
    """Fetch the coord-bearing child care rows from the NYC Open Data dataset (raises httpx.HTTPError
    on a bad status). Reuses the shared Socrata client, which returns the `:id` / `:updated_at`
    system fields so each row is addressable and carries its own 'as of' date."""
    return await query_dataset(CHILDCARE_DATASET, where=WHERE_HAS_COORDS, limit=limit, client=client)


# --- the tool --------------------------------------------------------------

def _site_citation(ctx: ToolContext, site: ChildCareSite, *,
                   origin_lat: float, origin_lon: float, dist_mi: float) -> str:
    """Register a row-addressed DATA citation: the single-row Socrata permalink, the row snapshot +
    content hash, and the distance derivation (so the eval floor can recompute it)."""
    url = row_url(CHILDCARE_DATASET, site.row_id) if site.row_id else dataset_url(CHILDCARE_DATASET)
    provenance = data_provenance(
        site.raw,
        record_id=site.row_id,
        field_pointer="/",
        derivation={"origin": [origin_lat, origin_lon], "point": [site.lat, site.lon],
                    "distance_mi": dist_mi},
    )
    return ctx.citations.register(
        url,
        snippet=f"{site.name} - {site.address}",
        title="Active NYC Health Code Regulated Child Care Programs (NYC Health Dept)",
        kind="DATA",
        valid_as_of=site.valid_as_of,
        provenance=provenance,
    )


def _descriptor(site: ChildCareSite) -> str:
    """A short grounded descriptor, e.g. '(preschool group child care)', from source-verified labels."""
    parts = [p for p in (site.program_label, site.facility_label) if p]
    return f" ({' '.join(parts)})" if parts else ""


def _site_block(site: ChildCareSite, cite: str, dist_mi: float) -> str:
    parts = [f"- {site.name}{_descriptor(site)} ({site.address or 'NYC'}) - "
             f"{dist_mi:.2f} mi straight-line {{cite:{cite}}}"]
    if site.age_range:
        parts.append(f"  Ages served: {site.age_range}")
    if site.capacity_phrase:
        parts.append(f"  Capacity: {site.capacity_phrase}")
    if site.administer_medication.lower() == "yes":
        parts.append("  Has staff qualified to administer medication")
    if site.phone:
        parts.append(f"  Phone: {site.phone} - call to check openings, hours, ages, and cost")
    parts.append(f"  Directions: {directions_link(site.lat, site.lon)}")
    parts.append(f"  As of: {site.valid_as_of or 'Source date unavailable'}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    if not near:
        return ("Ask the user where they are (an NYC address or neighborhood) before searching - "
                "never guess a child care program's location.")

    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return (f"I couldn't locate '{near}' in NYC, so I can't find nearby child care. Ask the "
                f"user for a specific NYC address or neighborhood - don't guess a program.")
    if origin.low_confidence:
        return _clarify_message(near)

    try:
        records = await _query_childcare(ctx.http)
    except httpx.HTTPError:
        return (f"I couldn't reach the NYC child care data right now - don't guess a program. "
                f"Point the user to {OFFICIAL}.")

    sites = [s for s in (_to_site(r) for r in records) if s is not None]
    if not sites:
        return (f"No NYC child care programs came back from the Health Department data. Don't invent "
                f"one - point the user to {OFFICIAL}.")

    k = int(args.get("k") or 5)
    ordered = sorted(sites, key=lambda s: haversine_m(origin.lat, origin.lon, s.lat, s.lon))
    # Collapse duplicate rows for the same physical program (same name + coordinate).
    ranked: list[ChildCareSite] = []
    seen: set[tuple] = set()
    for site in ordered:
        key = (site.name.strip().lower(), round(site.lat, 5), round(site.lon, 5))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(site)
        if len(ranked) >= k:
            break

    lines = [
        f"Origin: {origin.label} ({origin.lat:.5f},{origin.lon:.5f})",
        _resolution_note(near, origin),
        "NYC child care programs from the DOHMH regulated child care list (powers NYC Child Care "
        "Connect) - report only these, cite each:",
    ]
    for site in ranked:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, site.lat, site.lon))
        cite = _site_citation(ctx, site, origin_lat=origin.lat, origin_lon=origin.lon,
                              dist_mi=dist_mi)
        lines.append(_site_block(site, cite, dist_mi))
    lines.append("Capacity is each program's MAXIMUM licensed size, NOT open spots - a program may be "
                 "full. This data has NO hours, NO cost/tuition, and NO current openings: tell the "
                 "user to CALL each program to confirm availability, hours, ages, and cost. These are "
                 "DOHMH-permitted center-based programs (a health permit is not an endorsement); the "
                 "list does NOT include home/family child care or after-school (NY State OCFS), and it "
                 "does NOT show free 3-K/Pre-K seats or child care vouchers - point to MySchools.nyc "
                 "and ACCESS NYC for those, and don't assert a child qualifies from this data.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="nearest_child_care",
            description=(
                "Find the nearest NYC regulated child care programs (day care / preschool) to an "
                "address, grounded in the DOHMH 'Active NYC Health Code Regulated Child Care "
                "Programs' list (the data behind NYC Child Care Connect). Pass `near` = the user's "
                "NYC address or neighborhood; optional `k` (default 5). Returns each program's name, "
                "full address, phone, the age range served, the maximum licensed capacity, and the "
                "facility/program type - every program cited. NEVER guess a program: if geocoding "
                "fails or none are near, say so and point to NYC Child Care Connect / 311. Capacity "
                "is a MAX, not open spots; the source has NO hours, cost, or availability - tell the "
                "user to call. It does NOT cover home/family child care or free 3-K/Pre-K seats."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string",
                             "description": "The NYC address or neighborhood to search near."},
                    "k": {"type": "integer",
                          "description": "How many programs to return (default 5).", "default": 5},
                },
                "required": ["near"],
            },
            handler=_handler,
            open_world=True,  # hits the live NYC Open Data Socrata dataset + geocoder
        )
    ]
