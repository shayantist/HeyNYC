"""wic module tool: `nearest_wic_site`, grounded in the NY State WIC site directory.

Data source: the public, tokenless Socrata dataset that powers Health Data NY's "Women, Infants,
and Children (WIC) Program Site Information" map (`g4i5-r6zx` on health.data.ny.gov). WIC is
state-administered, so the authoritative site list is the state's, not a city dataset. We fetch the
NYC rows (the five boroughs served), rank them by Haversine distance from the user's geocoded
location (reused geo machinery), and return the closest few with: agency name, full address,
phone, website when listed, and the site type (Permanent vs. Temporary/rotating). Every site is a
row-addressed DATA citation resolving to its Socrata row permalink.

Honest limitations (enforced in the manifest prompt too): the source has NO hours field and NO
appointment info - we never invent hours; we tell the user to call. It also carries no eligibility
detail, so we never assert WIC eligibility from this data. A "Temporary" site rotates and may not
always be open, so we flag it and say to call ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import httpx

from heynyc.core import config
from heynyc.core.citations import data_provenance
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    _clarify_message,
    _resolution_note,
    geocode,
    haversine_m,
    miles,
)

# The live backend of Health Data NY's WIC Program Site Information map - verified public + tokenless.
WIC_DATASET = "g4i5-r6zx"
WIC_HOST = "https://health.data.ny.gov"
WIC_URL = f"{WIC_HOST}/resource/{WIC_DATASET}.json"
# NYC scope: the five boroughs, by the dataset's `counties_boroughs_served` labels (Kings=Brooklyn,
# Richmond=Staten Island, New York=Manhattan). Every NYC physical site carries one of these.
WHERE_NYC = "counties_boroughs_served in('Bronx','Kings','New York','Queens','Richmond')"
OFFICIAL = "the NY State WIC info at health.ny.gov/prevention/nutrition/wic or call 311"


def _clean(value) -> str:
    """None / literal 'NULL' / blanks → ''. Socrata omits empty fields, but be defensive."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() == "NULL" else text


def _row_permalink(row_id: str) -> str:
    """A single-row Socrata permalink: /resource/{4x4}/{:id}.json - a real, resolvable URL that
    returns exactly the cited row, so the DATA citation can be re-fetched and verified."""
    return f"{WIC_HOST}/resource/{WIC_DATASET}/{row_id}.json"


def _parse_location(record: dict) -> tuple[float, float] | None:
    """(lat, lon) from the Socrata `location` field `location_1`, or None if absent/unparseable.

    A Socrata `location` value is a dict with string `latitude`/`longitude` subfields. We never
    guess a coordinate: a missing or malformed location drops the row from the results."""
    loc = record.get("location_1")
    if not isinstance(loc, dict):
        return None
    try:
        return float(loc["latitude"]), float(loc["longitude"])
    except (KeyError, TypeError, ValueError):
        return None


def _address(record: dict) -> str:
    """Assemble the street address from the source's parts (any of which may be blank)."""
    line1 = ", ".join(p for p in (_clean(record.get("street_address")),
                                  _clean(record.get("street2"))) if p)
    tail = " ".join(p for p in (_clean(record.get("city")),
                                _clean(record.get("state")),
                                _clean(record.get("zip"))) if p)
    return ", ".join(p for p in (line1, tail) if p)


def _website(record: dict) -> str:
    """The site's website URL from the Socrata `url` field `link_to_website` (a dict with `url`)."""
    link = record.get("link_to_website")
    if isinstance(link, dict):
        return _clean(link.get("url"))
    return ""


def _valid_as_of(record: dict) -> str:
    """The row's Socrata `:updated_at` change signal, or blank when unavailable."""
    text = _clean(record.get(":updated_at"))
    if text:
        try:
            return datetime.fromisoformat(text[:10]).date().isoformat()
        except ValueError:
            pass
    return ""


# --- record → site ---------------------------------------------------------

@dataclass
class WicSite:
    name: str
    lat: float
    lon: float
    address: str
    phone: str
    website: str
    site_type: str
    site_number: str
    borough: str
    row_id: str
    valid_as_of: str
    raw: dict = field(default_factory=dict)


def _to_site(record: dict) -> WicSite | None:
    """Map a raw Socrata record to a WicSite; drop records without usable coordinates."""
    coords = _parse_location(record)
    if coords is None:
        return None
    lat, lon = coords
    return WicSite(
        name=_clean(record.get("agency_name")) or "WIC site",
        lat=lat,
        lon=lon,
        address=_address(record),
        phone=_clean(record.get("phone_number")),
        website=_website(record),
        site_type=_clean(record.get("site_type")),
        site_number=_clean(record.get("site_number")),
        borough=_clean(record.get("counties_boroughs_served")),
        row_id=_clean(record.get(":id")),
        valid_as_of=_valid_as_of(record),
        raw=record,
    )


def directions_link(lat: float, lon: float) -> str:
    """A Google Maps directions deep link to a grounded coordinate (navigation handoff, no citation
    needed - it's a deterministic transform of an already-grounded point)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.5f},{lon:.5f}"


async def _query_wic(client: httpx.AsyncClient, *, where: str, limit: int = 500) -> list[dict]:
    """Fetch WIC site rows from the Health Data NY Socrata dataset (raises httpx.HTTPError on a bad
    status). Points at health.data.ny.gov, not the NYC SOCRATA_BASE, so we build the request here
    rather than reuse `datasets.query_dataset`; `$$exclude_system_fields=false` returns the `:id` /
    `:updated_at` fields so each row is addressable and carries its own 'as of' date."""
    params: dict = {"$where": where, "$limit": limit, "$$exclude_system_fields": "false"}
    headers: dict = {}
    if config.SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = config.SOCRATA_APP_TOKEN
    response = await client.get(WIC_URL, params=params, headers=headers)
    response.raise_for_status()
    return response.json() or []


# --- the tool --------------------------------------------------------------

def _site_citation(ctx: ToolContext, site: WicSite, *,
                   origin_lat: float, origin_lon: float, dist_mi: float) -> str:
    """Register a row-addressed DATA citation: the single-row Socrata permalink, the row snapshot +
    content hash, and the distance derivation (so the eval floor can recompute it)."""
    url = _row_permalink(site.row_id) if site.row_id else WIC_URL
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
        title="NY State WIC Program Site Information (Health Data NY)",
        kind="DATA",
        valid_as_of=site.valid_as_of,
        provenance=provenance,
    )


def _site_block(site: WicSite, cite: str, dist_mi: float) -> str:
    temp = " (temporary/rotating site - call to confirm it's open)" if \
        site.site_type.lower() == "temporary" else ""
    parts = [f"- {site.name}{temp} ({site.address or 'NYC'}) - "
             f"{dist_mi:.2f} mi straight-line {{cite:{cite}}}"]
    if site.phone:
        parts.append(f"  Phone: {site.phone} - call for hours and to book an appointment")
    if site.website:
        parts.append(f"  Website: {site.website}")
    parts.append(f"  Directions: {directions_link(site.lat, site.lon)}")
    parts.append(f"  As of: {site.valid_as_of or 'Source date unavailable'}")
    return "\n".join(parts)


async def _handler(args: dict, ctx: ToolContext) -> str:
    near = (args.get("near") or "").strip()
    if not near:
        return ("Ask the user where they are (an NYC address or neighborhood) before searching - "
                "never guess a WIC site location.")

    origin = await geocode(near, client=ctx.http)
    if origin is None:
        return (f"I couldn't locate '{near}' in NYC, so I can't find a nearby WIC site. Ask the "
                f"user for a specific NYC address or neighborhood - don't guess a site.")
    if origin.low_confidence:
        return _clarify_message(near)

    try:
        records = await _query_wic(ctx.http, where=WHERE_NYC)
    except httpx.HTTPError:
        return (f"I couldn't reach the NY State WIC site data right now - don't guess a WIC site. "
                f"Point the user to {OFFICIAL}.")

    sites = [s for s in (_to_site(r) for r in records) if s is not None]
    if not sites:
        return (f"No NYC WIC sites came back from the NY State WIC data. Don't invent one - "
                f"point the user to {OFFICIAL}.")

    k = int(args.get("k") or 5)
    ordered = sorted(sites, key=lambda s: haversine_m(origin.lat, origin.lon, s.lat, s.lon))
    # Collapse duplicate rows for the same physical site (same name + coordinate).
    ranked: list[WicSite] = []
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
        "NYC WIC sites from NY State WIC Program Site Information (Health Data NY) - report only "
        "these, cite each:",
    ]
    for site in ranked:
        dist_mi = miles(haversine_m(origin.lat, origin.lon, site.lat, site.lon))
        cite = _site_citation(ctx, site, origin_lat=origin.lat, origin_lon=origin.lon,
                              dist_mi=dist_mi)
        lines.append(_site_block(site, cite, dist_mi))
    lines.append("This data has NO hours and NO appointment info - tell the user to call the site "
                 "for hours and to book. WIC has income and category rules (pregnant, postpartum, "
                 "infants, and children under 5); don't assert eligibility from this list - point "
                 "to health.ny.gov/prevention/nutrition/wic to apply and check eligibility.")
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="nearest_wic_site",
            description=(
                "Find the nearest NYC WIC (Women, Infants, and Children) sites to an address, "
                "grounded in the NY State WIC Program Site Information directory (Health Data NY). "
                "Pass `near` = the user's NYC address or neighborhood; optional `k` (default 5). "
                "Returns each site's agency name, full address, phone, website if listed, and site "
                "type (Permanent vs. Temporary) - every site cited. NEVER guess a site: if geocoding "
                "fails or none are near, say so and point to the state WIC info / 311. The source has "
                "NO hours and NO eligibility detail - tell the user to call; don't invent either."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "near": {"type": "string",
                             "description": "The NYC address or neighborhood to search near."},
                    "k": {"type": "integer",
                          "description": "How many WIC sites to return (default 5).", "default": 5},
                },
                "required": ["near"],
            },
            handler=_handler,
            open_world=True,  # hits the live Health Data NY Socrata dataset + geocoder
        )
    ]
