"""childcare module tool: `find_child_care_connect_programs`, grounded in the DOHMH regulated child care list.

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
from datetime import date, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.datasets import dataset_url, query_dataset, row_url
from heynyc.core.tools.geo import (
    GeoPoint,
    miles,
    origin_precision,
    rank_nearby,
    resolve_location,
    resolved_location_citation,
)

# The live backend of NYC Child Care Connect - verified public + tokenless on data.cityofnewyork.us.
CHILDCARE_DATASET = "gy3q-4tzp"
# Only rows with a usable coordinate (the source has a handful with none); we never guess a location.
WHERE_HAS_COORDS = "latitude IS NOT NULL AND longitude IS NOT NULL"
CHILDCARE_CONTACT_URL = "https://www.nyc.gov/site/doh/services/child-care.page"
CHILDCARE_PAGE_SIZE = 1000

# Verified against the dataset's own column descriptions (DOHMH): facility_type is GCC (group child
# care, Article 47) or SBCC (school-based child care, Article 43). We only ever surface a label we
# can ground in the source code; an unknown code yields no label rather than an invented one.
_FACILITY_LABELS = {"GCC": "group child care", "SBCC": "school-based child care"}
_PROGRAM_LABELS = {"PRESCHOOL": "preschool", "INFANT TODDLER": "infant/toddler"}


class ChildCareQuery(LocationRequest):
    near: str = Field(description="NYC address or neighborhood to search near.")
    max_results: int | None = Field(
        default=None, ge=1, description="Maximum programs requested; omit for the default 5."
    )


class ChildCareSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_called", "ok", "partial", "unavailable"]
    url: str = dataset_url(CHILDCARE_DATASET)
    fetched_at: datetime | None = None
    returned_count: int | None = None
    usable_count: int | None = None
    complete: bool | None = None
    pages_fetched: int = 0
    page_size: int = CHILDCARE_PAGE_SIZE
    next_offset: int | None = None
    query_filter: str = WHERE_HAS_COORDS
    includes_rows_without_coordinates: Literal[False] = False
    excluded_row_count: int | None = None
    error: Literal["transport_error", "invalid_response"] | None = None


class ChildCareOrigin(BaseModel):
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


class ChildCareOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class ChildCareService(BaseModel):
    model_config = ConfigDict(extra="forbid")

    program_type: str | None = None
    age_range: str | None = None
    licensed_capacity: int | None = None
    current_openings: None = None
    cost: None = None
    language: None = None
    accessibility: None = None
    service_area: None = None
    eligibility: None = None
    required_document: None = None


class ChildCareLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    street_address: str | None = None
    borough: str | None = None
    postal_code: str | None = None
    latitude: float
    longitude: float


class ChildCarePhone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str
    type: Literal["voice"] = "voice"


class ChildCareServiceAtLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_type: str | None = None
    can_administer_medication: bool | None = None
    schedule: None = None


class ChildCareProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_record_id: str
    program_id: str | None = None
    permit_number: str | None = None
    organization: ChildCareOrganization
    service: ChildCareService
    location: ChildCareLocation
    service_at_location: ChildCareServiceAtLocation
    phone: ChildCarePhone | None = None
    distance_miles: float
    valid_as_of: date | None = None
    citation_id: str
    action_url: str


class ChildCareDirectoryScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    covers_center_based_programs: Literal[True] = True
    covers_home_based_programs: Literal[False] = False
    covers_after_school_programs: Literal[False] = False
    includes_hours: Literal[False] = False
    includes_cost: Literal[False] = False
    includes_current_openings: Literal[False] = False
    licensed_capacity_means_open_spots: Literal[False] = False
    health_permit_is_endorsement: Literal[False] = False


class ChildCareDirectoryRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = CHILDCARE_CONTACT_URL
    citation_id: str


ChildCareOutcome = Literal[
    "success",
    "missing_origin",
    "location_not_found",
    "location_ambiguous",
    "source_partial",
    "source_unavailable",
    "no_results",
]


class ChildCareResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: ChildCareOutcome
    origin: ChildCareOrigin | None = None
    origin_citation_id: str | None = None
    primary_citation_id: str | None = None
    source: ChildCareSource
    scope: ChildCareDirectoryScope = Field(default_factory=ChildCareDirectoryScope)
    programs: list[ChildCareProgram] = Field(default_factory=list)
    directory_route: ChildCareDirectoryRoute | None = None
    requested_count: int | None = None


class _ChildCareQueryPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict]
    pages_fetched: int
    complete: bool
    error: Literal["transport_error", "invalid_response"] | None = None


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


def _capacity(record: dict) -> int | None:
    cap = _clean(record.get("capacity"))
    if not cap:
        return None
    try:
        return int(cap)
    except ValueError:
        return None


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
    licensed_capacity: int | None
    facility_label: str
    program_label: str
    can_administer_medication: bool | None
    program_id: str
    permit_number: str
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
        licensed_capacity=_capacity(record),
        facility_label=_facility_label(record),
        program_label=_program_label(record),
        can_administer_medication=(
            True if _clean(record.get("administer_medication")).casefold() == "yes"
            else False if _clean(record.get("administer_medication")).casefold() == "no"
            else None
        ),
        program_id=_clean(record.get("dcid")),
        permit_number=_clean(record.get("permit_number")),
        row_id=_clean(record.get(":id")),
        valid_as_of=_valid_as_of(record),
        raw=record,
    )


def directions_link(lat: float, lon: float) -> str:
    """A Google Maps directions deep link to a grounded coordinate (navigation handoff, no citation
    needed - it's a deterministic transform of an already-grounded point)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.5f},{lon:.5f}"


async def _query_childcare(
    client: httpx.AsyncClient,
    *,
    page_size: int = CHILDCARE_PAGE_SIZE,
) -> _ChildCareQueryPage:
    """Read every stable Socrata page needed for correct citywide distance ranking."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    pages_fetched = 0
    while True:
        try:
            page = await query_dataset(
                CHILDCARE_DATASET,
                where=WHERE_HAS_COORDS,
                order=":id",
                limit=page_size,
                offset=len(rows),
                client=client,
            )
            if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
                raise ValueError("Socrata child care response is not a row list")
        except httpx.HTTPError:
            if not rows:
                raise
            return _ChildCareQueryPage(
                rows=rows,
                pages_fetched=pages_fetched,
                complete=False,
                error="transport_error",
            )
        except (ValueError, TypeError, AttributeError):
            if not rows:
                raise
            return _ChildCareQueryPage(
                rows=rows,
                pages_fetched=pages_fetched,
                complete=False,
                error="invalid_response",
            )
        if not page:
            return _ChildCareQueryPage(
                rows=rows,
                pages_fetched=pages_fetched,
                complete=True,
            )
        page_ids = [str(row.get(":id") or "") for row in page]
        if (
            any(not row_id for row_id in page_ids)
            or len(set(page_ids)) != len(page_ids)
            or seen_ids.intersection(page_ids)
        ):
            if rows:
                return _ChildCareQueryPage(
                    rows=rows,
                    pages_fetched=pages_fetched,
                    complete=False,
                    error="invalid_response",
                )
            raise ValueError("Socrata child care page lacks stable unique row IDs")
        seen_ids.update(page_ids)
        rows.extend(page)
        pages_fetched += 1


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


def _result(
    outcome: ChildCareOutcome,
    *,
    source_status: Literal["not_called", "ok", "unavailable"] = "not_called",
    source_error: Literal["transport_error", "invalid_response"] | None = None,
    source_fetched_at: datetime | None = None,
    **updates,
) -> ChildCareResult:
    return ChildCareResult(
        outcome=outcome,
        source=ChildCareSource(
            status=source_status,
            fetched_at=source_fetched_at,
            error=(
                source_error
                or ("transport_error" if source_status == "unavailable" else None)
            ),
        ),
        **updates,
    )


def _origin_result(point: GeoPoint, query: str) -> ChildCareOrigin:
    return ChildCareOrigin(
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


def _directory_route(ctx: ToolContext) -> ChildCareDirectoryRoute:
    citation_id = ctx.citations.register(
        CHILDCARE_CONTACT_URL,
        snippet="Find permitted child care programs in NYC Child Care Connect",
        title="NYC Health: Child Care Connect",
        kind="DOC",
    )
    return ChildCareDirectoryRoute(citation_id=citation_id)


def _source_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


async def _handler(args: ChildCareQuery, ctx: ToolContext) -> ChildCareResult:
    query = ChildCareQuery.model_validate(args)
    near = query.near.strip()
    if not near:
        return _result("missing_origin")

    origin = await resolve_location(near, ctx)
    if origin is None:
        return _result(
            "location_not_found",
            directory_route=_directory_route(ctx),
        )
    if origin.low_confidence:
        return _result(
            "location_ambiguous",
            origin=_origin_result(origin, near),
            directory_route=_directory_route(ctx),
        )
    if origin.resident_query:
        ctx.current_location = origin

    fetched_at = datetime.now().astimezone()
    try:
        page = await _query_childcare(ctx.http)
    except httpx.HTTPError:
        return _result(
            "source_unavailable",
            source_status="unavailable",
            source_fetched_at=fetched_at,
            origin=_origin_result(origin, near),
            directory_route=_directory_route(ctx),
        )
    except (ValueError, TypeError, AttributeError):
        return _result(
            "source_unavailable",
            source_status="unavailable",
            source_error="invalid_response",
            source_fetched_at=fetched_at,
            origin=_origin_result(origin, near),
            directory_route=_directory_route(ctx),
        )

    sites = [s for s in (_to_site(row) for row in page.rows) if s is not None]
    source_complete = page.complete and len(sites) == len(page.rows)
    if not sites:
        return ChildCareResult(
            outcome="no_results" if source_complete else "source_partial",
            origin=_origin_result(origin, near),
            origin_citation_id=resolved_location_citation(ctx, origin),
            source=ChildCareSource(
                status="ok" if source_complete else "partial",
                fetched_at=fetched_at,
                returned_count=len(page.rows),
                usable_count=0,
                complete=source_complete,
                pages_fetched=page.pages_fetched,
                next_offset=None if source_complete else len(page.rows),
                excluded_row_count=len(page.rows),
                error=page.error,
            ),
            directory_route=_directory_route(ctx),
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

    programs: list[ChildCareProgram] = []
    for site, distance_m in ranked:
        dist_mi = miles(distance_m)
        citation_id = _site_citation(
            ctx,
            site,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            dist_mi=dist_mi,
        )
        programs.append(ChildCareProgram(
            provider_record_id=site.row_id,
            program_id=site.program_id or None,
            permit_number=site.permit_number or None,
            organization=ChildCareOrganization(name=site.name),
            service=ChildCareService(
                program_type=site.program_label or None,
                age_range=site.age_range or None,
                licensed_capacity=site.licensed_capacity,
            ),
            location=ChildCareLocation(
                address=site.address or None,
                street_address=_clean(site.raw.get("address")) or None,
                borough=_clean(site.raw.get("borough")) or None,
                postal_code=_clean(site.raw.get("zipcode")) or None,
                latitude=site.lat,
                longitude=site.lon,
            ),
            service_at_location=ChildCareServiceAtLocation(
                facility_type=site.facility_label or None,
                can_administer_medication=site.can_administer_medication,
            ),
            phone=ChildCarePhone(number=site.phone) if site.phone else None,
            distance_miles=dist_mi,
            valid_as_of=_source_date(site.valid_as_of),
            citation_id=citation_id,
            action_url=directions_link(site.lat, site.lon),
        ))
    return ChildCareResult(
        outcome="success" if source_complete else "source_partial",
        origin=_origin_result(origin, near),
        origin_citation_id=resolved_location_citation(ctx, origin),
        primary_citation_id=programs[0].citation_id,
        source=ChildCareSource(
            status="ok" if source_complete else "partial",
            fetched_at=fetched_at,
            returned_count=len(page.rows),
            usable_count=len(sites),
            complete=source_complete,
            pages_fetched=page.pages_fetched,
            next_offset=None if source_complete else len(page.rows),
            excluded_row_count=len(page.rows) - len(sites),
            error=page.error,
        ),
        programs=programs,
        directory_route=_directory_route(ctx),
        requested_count=max_results,
    )


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_child_care_connect_programs",
            description=(
                "Find the nearest NYC regulated child care programs (day care / preschool) to an "
                "address, grounded in the DOHMH 'Active NYC Health Code Regulated Child Care "
                "Programs' list (the data behind NYC Child Care Connect). Pass `near` = the user's "
                "NYC address or neighborhood; optional `max_results` (default 5). Returns each program's name, "
                "full address, phone, the age range served, the maximum licensed capacity, and the "
                "facility/program type - every program cited. NEVER guess a program: if geocoding "
                "fails or none are near, say so and point to NYC Child Care Connect / 311. Capacity "
                "is a MAX, not open spots; the source has NO hours, cost, or availability - tell the "
                "user to call. It does NOT cover home/family child care or free 3-K/Pre-K seats."
            ),
            input_type=ChildCareQuery,
            handler=_handler,
            return_type=ChildCareResult,
            open_world=True,  # hits the live NYC Open Data Socrata dataset + geocoder
        )
    ]
