"""WIC site finder grounded in the NY State WIC site directory.

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
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field

from heynyc.core import config
from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    miles,
    origin_precision,
    rank_nearby,
    resolve_location,
    resolved_location_citation,
)

# The live backend of Health Data NY's WIC Program Site Information map - verified public + tokenless.
WIC_DATASET = "g4i5-r6zx"
WIC_HOST = "https://health.data.ny.gov"
WIC_URL = f"{WIC_HOST}/resource/{WIC_DATASET}.json"
# NYC scope: the five boroughs, by the dataset's `counties_boroughs_served` labels (Kings=Brooklyn,
# Richmond=Staten Island, New York=Manhattan). Every NYC physical site carries one of these.
WHERE_NYC = "counties_boroughs_served in('Bronx','Kings','New York','Queens','Richmond')"
WIC_APPLY_URL = "https://www.health.ny.gov/prevention/nutrition/wic/how_to_apply.htm"
WIC_LIMIT = 500
WIC_QUERY_URL = f"{WIC_URL}?" + urlencode({
    "$where": WHERE_NYC,
    "$limit": WIC_LIMIT,
    "$order": ":id",
    "$$exclude_system_fields": "false",
})


class WicQuery(LocationRequest):
    near: str = Field(description="NYC address or neighborhood to search near.")
    max_results: int | None = Field(
        default=None, ge=1, le=10, description="Maximum WIC sites requested; omit for the default 5."
    )


class WicSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_called", "ok", "partial", "unavailable"]
    url: str = WIC_QUERY_URL
    fetched_at: datetime | None = None
    returned_count: int | None = None
    usable_count: int | None = None
    complete: bool | None = None
    pages_fetched: int = 0
    page_size: int = WIC_LIMIT
    next_offset: int | None = None
    error: Literal["transport_error", "invalid_response"] | None = None


class WicOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resident_query: str
    label: str
    latitude: float
    longitude: float
    match_type: str | None = None
    provider_id: str | None = None
    confidence: float | None = None
    low_confidence: bool
    precision: Literal["precise", "approximate"]


class WicOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str


class WicService(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: Literal["WIC"] = "WIC"
    eligibility_description: None = None
    required_document: None = None
    language: None = None


class WicLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    physical_address: str | None = None
    latitude: float
    longitude: float
    borough: str | None = None
    accessibility: None = None


class WicServiceAtLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    service_id: str
    location_id: str
    site_number: str | None = None
    site_type: str | None = None


class WicPhone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str


class WicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization: WicOrganization
    service: WicService
    location: WicLocation
    service_at_location: WicServiceAtLocation
    phone: WicPhone | None = None
    website: str | None = None
    hours: None = None
    appointment_required: None = None
    distance_miles: float
    distance_method: Literal["haversine"] = "haversine"
    origin_precision: Literal["precise", "approximate"]
    valid_as_of: date | None = None
    citation_id: str
    action_url: str


class WicApplicationRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = WIC_APPLY_URL
    citation_id: str


WicOutcome = Literal[
    "success",
    "missing_origin",
    "location_not_found",
    "location_ambiguous",
    "source_partial",
    "source_unavailable",
    "no_results",
]


class WicResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: WicOutcome
    origin: WicOrigin | None = None
    origin_citation_id: str | None = None
    primary_citation_id: str | None = None
    source: WicSource
    records: list[WicRecord] = Field(default_factory=list)
    application_route: WicApplicationRoute | None = None
    requested_count: int | None = None


class _WicQueryPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict]
    pages_fetched: int
    complete: bool
    error: Literal["transport_error", "invalid_response"] | None = None


def _clean(value) -> str:
    """None / literal 'NULL' / blanks → ''. Socrata omits empty fields, but be defensive."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() == "NULL" else text


def _result(
    outcome: WicOutcome,
    *,
    source_status: Literal["not_called", "ok", "unavailable"] = "not_called",
    source_error: Literal["transport_error", "invalid_response"] | None = None,
    source_fetched_at: datetime | None = None,
    **updates,
) -> WicResult:
    return WicResult(
        outcome=outcome,
        source=WicSource(
            status=source_status,
            fetched_at=source_fetched_at,
            error=(
                source_error
                or ("transport_error" if source_status == "unavailable" else None)
            ),
        ),
        **updates,
    )


def _origin_result(point: GeoPoint, query: str) -> WicOrigin:
    return WicOrigin(
        resident_query=point.resident_query or query,
        label=point.label,
        latitude=point.lat,
        longitude=point.lon,
        match_type=point.match_type or None,
        provider_id=point.provider_id or None,
        confidence=point.confidence,
        low_confidence=point.low_confidence,
        precision=origin_precision(query, point),
    )


def _application_route(ctx: ToolContext) -> WicApplicationRoute:
    citation_id = ctx.citations.register(
        WIC_APPLY_URL,
        snippet="Apply or recertify for WIC by contacting a local WIC office",
        title="Apply or Recertify for WIC, New York State Department of Health",
        kind="DOC",
    )
    return WicApplicationRoute(citation_id=citation_id)


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


def _phone(record: dict) -> str:
    """Render the dataset's `base+extension` convention without inventing a second phone number."""
    phone = _clean(record.get("phone_number"))
    base, separator, extension = phone.rpartition("+")
    if separator and len("".join(filter(str.isdigit, base))) == 10 and extension.isdigit():
        return f"{base} ext. {extension}"
    return phone


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
        phone=_phone(record),
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


async def _query_wic(
    client: httpx.AsyncClient | None,
    *,
    where: str,
    limit: int = WIC_LIMIT,
) -> _WicQueryPage:
    """Read every stable Health Data NY Socrata page needed for citywide distance ranking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    headers: dict = {}
    if config.SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = config.SOCRATA_APP_TOKEN
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=20.0)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    pages_fetched = 0
    try:
        while True:
            try:
                response = await client.get(
                    WIC_URL,
                    params={
                        "$where": where,
                        "$limit": limit,
                        "$offset": len(rows),
                        "$order": ":id",
                        "$$exclude_system_fields": "false",
                    },
                    headers=headers,
                )
                response.raise_for_status()
                page = response.json()
                if not isinstance(page, list) or any(
                    not isinstance(record, dict) for record in page
                ):
                    raise ValueError("WIC Socrata response must be a list of objects")
            except httpx.HTTPError:
                if not rows:
                    raise
                return _WicQueryPage(
                    rows=rows,
                    pages_fetched=pages_fetched,
                    complete=False,
                    error="transport_error",
                )
            except (ValueError, TypeError, AttributeError):
                if not rows:
                    raise
                return _WicQueryPage(
                    rows=rows,
                    pages_fetched=pages_fetched,
                    complete=False,
                    error="invalid_response",
                )
            if not page:
                return _WicQueryPage(
                    rows=rows,
                    pages_fetched=pages_fetched,
                    complete=True,
                )
            page_ids = [str(record.get(":id") or "") for record in page]
            if (
                any(not row_id for row_id in page_ids)
                or len(set(page_ids)) != len(page_ids)
                or seen_ids.intersection(page_ids)
            ):
                if not rows:
                    raise ValueError("WIC Socrata page lacks stable unique row IDs")
                return _WicQueryPage(
                    rows=rows,
                    pages_fetched=pages_fetched,
                    complete=False,
                    error="invalid_response",
                )
            seen_ids.update(page_ids)
            rows.extend(page)
            pages_fetched += 1
    finally:
        if own_client:
            await client.aclose()


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


async def _handler(args: dict, ctx: ToolContext) -> WicResult:
    query = WicQuery.model_validate(args)
    near = query.near.strip()
    if not near:
        return _result("missing_origin")

    origin = await resolve_location(near, ctx)
    if origin is None:
        return _result(
            "location_not_found",
            application_route=_application_route(ctx),
        )
    if origin.low_confidence:
        return _result(
            "location_ambiguous",
            origin=_origin_result(origin, near),
            application_route=_application_route(ctx),
        )
    if origin.resident_query:
        ctx.current_location = origin

    fetched_at = datetime.now().astimezone()
    try:
        page = await _query_wic(ctx.http, where=WHERE_NYC)
    except httpx.HTTPError:
        return _result(
            "source_unavailable",
            source_status="unavailable",
            source_fetched_at=fetched_at,
            origin=_origin_result(origin, near),
            application_route=_application_route(ctx),
        )
    except (ValueError, TypeError, AttributeError):
        return _result(
            "source_unavailable",
            source_status="unavailable",
            source_error="invalid_response",
            source_fetched_at=fetched_at,
            origin=_origin_result(origin, near),
            application_route=_application_route(ctx),
        )

    sites = [s for s in (_to_site(row) for row in page.rows) if s is not None]
    if not page.complete:
        return WicResult(
            outcome="source_partial",
            origin=_origin_result(origin, near),
            origin_citation_id=resolved_location_citation(ctx, origin),
            source=WicSource(
                status="partial",
                fetched_at=fetched_at,
                returned_count=len(page.rows),
                usable_count=len(sites),
                complete=False,
                pages_fetched=page.pages_fetched,
                next_offset=len(page.rows),
                error=page.error,
            ),
            application_route=_application_route(ctx),
        )
    if not sites:
        return WicResult(
            outcome="no_results",
            origin=_origin_result(origin, near),
            origin_citation_id=resolved_location_citation(ctx, origin),
            source=WicSource(
                status="ok",
                fetched_at=fetched_at,
                returned_count=len(page.rows),
                usable_count=0,
                complete=page.complete,
                pages_fetched=page.pages_fetched,
            ),
            application_route=_application_route(ctx),
        )

    max_results = query.max_results or 5
    ranked = rank_nearby(
        origin,
        sites,
        key=lambda site: (
            site.name.strip().casefold(),
            round(site.lat, 5),
            round(site.lon, 5),
        ),
        limit=max_results,
    )
    application_route = _application_route(ctx)
    typed_records = []
    for site, distance_m in ranked:
        dist_mi = miles(distance_m)
        cite = _site_citation(
            ctx,
            site,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            dist_mi=dist_mi,
        )
        record_id = site.row_id or f"{site.lat:.5f},{site.lon:.5f}"
        service_id = f"{record_id}:service"
        location_id = f"{record_id}:location"
        typed_records.append(WicRecord(
            organization=WicOrganization(name=site.name),
            service=WicService(id=service_id),
            location=WicLocation(
                id=location_id,
                name=site.name,
                physical_address=site.address or None,
                latitude=site.lat,
                longitude=site.lon,
                borough=site.borough or None,
            ),
            service_at_location=WicServiceAtLocation(
                id=record_id,
                service_id=service_id,
                location_id=location_id,
                site_number=site.site_number or None,
                site_type=site.site_type or None,
            ),
            phone=WicPhone(number=site.phone) if site.phone else None,
            website=site.website or None,
            distance_miles=dist_mi,
            origin_precision=origin_precision(near, origin),
            valid_as_of=(
                date.fromisoformat(site.valid_as_of) if site.valid_as_of else None
            ),
            citation_id=cite,
            action_url=directions_link(site.lat, site.lon),
        ))
    return WicResult(
        outcome="success",
        origin=_origin_result(origin, near),
        origin_citation_id=resolved_location_citation(ctx, origin),
        primary_citation_id=typed_records[0].citation_id,
        source=WicSource(
            status="ok",
            fetched_at=fetched_at,
            returned_count=len(page.rows),
            usable_count=len(sites),
            complete=page.complete,
            pages_fetched=page.pages_fetched,
        ),
        records=typed_records,
        application_route=application_route,
        requested_count=max_results,
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_wic_sites",
            description=(
                "Find the nearest NYC WIC (Women, Infants, and Children) sites to an address, "
                "grounded in the NY State WIC Program Site Information directory (Health Data NY). "
                "Pass `near` = the user's NYC address or neighborhood; optional `max_results` (default 5). "
                "Returns each site's agency name, full address, phone, website if listed, and site "
                "type (Permanent vs. Temporary) - every site cited. NEVER guess a site: if geocoding "
                "fails or none are near, say so and point to the state WIC info / 311. The source has "
                "NO hours and NO eligibility detail - tell the user to call; don't invent either."
            ),
            parameters=WicQuery.model_json_schema(),
            handler=_handler,
            return_type=WicResult,
            open_world=True,  # hits the live Health Data NY Socrata dataset + geocoder
        )
    ]
