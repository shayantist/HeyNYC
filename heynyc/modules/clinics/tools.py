"""Clinics module tools for safety-net locations and grounded coverage guidance.
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
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from heynyc.core.citations import data_provenance
from heynyc.core.location import LocationRequest
from heynyc.core.tools.arcgis import feature_query_url, query_feature_service_page
from heynyc.core.tools.base import Tool, ToolContext
from heynyc.core.tools.geo import (
    GeoPoint,
    maps_link,
    miles,
    origin_precision,
    rank_nearby,
    resolve_location,
    resolved_location_citation,
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


class ClinicQuery(LocationRequest):
    near: str = Field(description="NYC address, neighborhood, or ZIP to search near.")
    max_results: int | None = Field(
        default=None, ge=1, le=10, description="Maximum clinics requested; omit for the default 5."
    )


class HrsaClinicSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_called", "ok", "partial", "unavailable"]
    url: str = HRSA_URL
    query_filter: str = FQHC_WHERE
    fetched_at: datetime | None = None
    returned_count: int | None = None
    usable_count: int | None = None
    complete: bool | None = None
    pages_fetched: int = 0
    next_offset: int | None = None
    pagination_token: str | None = None
    error: Literal["transport_error", "invalid_response"] | None = None


class NycCareClinicSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_called", "ok", "partial", "empty", "unavailable"]
    url: str = NYC_CARE_SOURCE
    returned_count: int
    usable_count: int
    complete: bool
    valid_as_of: date = date.fromisoformat(VERIFIED_ON)
    error: Literal["file_unavailable", "invalid_rows"] | None = None


class ClinicSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hrsa: HrsaClinicSource
    nyc_care: NycCareClinicSource


class ClinicOrigin(BaseModel):
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


class ClinicOrganization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class ClinicService(BaseModel):
    model_config = ConfigDict(extra="forbid")

    program_class: Literal["FQHC", "NYC_CARE"]
    program_label: str
    hours: None = None
    language: None = None
    accessibility: None = None
    service_area: None = None
    eligibility: None = None
    required_document: None = None
    cost: None = None


class ClinicLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    borough: str | None = None
    latitude: float
    longitude: float


class ClinicPhone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str
    type: Literal["voice"] = "voice"


class ClinicServiceAtLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule: None = None
    services: None = None


class ClinicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_record_id: str
    organization: ClinicOrganization
    service: ClinicService
    location: ClinicLocation
    service_at_location: ClinicServiceAtLocation
    phone: ClinicPhone | None = None
    website: str | None = None
    distance_miles: float
    valid_as_of: date | None = None
    citation_id: str
    action_url: str


class ClinicProgramGuarantee(BaseModel):
    model_config = ConfigDict(extra="forbid")

    program_class: Literal["FQHC", "NYC_CARE"]
    label: str
    verified_fact: str
    valid_as_of: date
    citation_id: str


class ClinicRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    phone: str | None = None
    citation_id: str


ClinicOutcome = Literal[
    "success",
    "degraded",
    "missing_origin",
    "location_not_found",
    "location_ambiguous",
    "source_unavailable",
]


class ClinicResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: ClinicOutcome
    origin: ClinicOrigin | None = None
    origin_citation_id: str | None = None
    primary_citation_id: str | None = None
    sources: ClinicSources
    clinics: list[ClinicRecord] = Field(default_factory=list)
    program_guarantees: list[ClinicProgramGuarantee] = Field(default_factory=list)
    fallback_routes: list[ClinicRoute] = Field(default_factory=list)
    requested_count: int | None = None


class _HrsaQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict]
    pages_fetched: int
    complete: bool
    next_offset: int | None = None
    pagination_token: str | None = None
    error: Literal["transport_error", "invalid_response"] | None = None


@dataclass(frozen=True)
class ProgramGuarantee:
    """A CLASS's grounded eligibility framing + the citations that back it.

    doc_url/doc_title/snippet/body -> the DOC citation to the official PROGRAM page (the eligibility
    guarantee). data_title -> the title for each facility's DATA citation (the facility source).
    """
    label: str
    doc_url: str       # official program page (verified live)
    doc_title: str
    snippet: str       # short cite label, subset of `body`
    body: str          # the grounded eligibility / cost / immigration-safety sentence(s), cited
    data_title: str    # citation title for the facility (DATA) source


CLASS_GUARANTEE: dict[str, ProgramGuarantee] = {
    CLASS_FQHC: ProgramGuarantee(
        label="Community Health Center (FQHC)",
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
    valid_as_of: str | None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _NycCareCatalog:
    clinics: list[Clinic]
    returned_count: int
    complete: bool
    error: Literal["file_unavailable", "invalid_rows"] | None = None


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
        valid_as_of=None,
        raw=record,
    )


# --- NYC_CARE (bundled, build-time-geocoded seed) --------------------------

def _load_nyc_care_seed(path: Path = SEED_PATH) -> _NycCareCatalog:
    """Load the bundled NYC Care / H+H seed (build-time geocoded). Rows without coords are dropped.

    A missing or partially invalid seed retains its distinct source outcome.
    """
    clinics: list[Clinic] = []
    returned_count = 0
    invalid_rows = False
    try:
        with path.open(encoding="utf-8") as fh:
            rows = csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t")
            for row in rows:
                returned_count += 1
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                except (KeyError, TypeError, ValueError):
                    invalid_rows = True
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
        return _NycCareCatalog(
            clinics=[],
            returned_count=0,
            complete=False,
            error="file_unavailable",
        )
    return _NycCareCatalog(
        clinics=clinics,
        returned_count=returned_count,
        complete=not invalid_rows,
        error="invalid_rows" if invalid_rows else None,
    )


# --- citations -------------------------------------------------------------

def _facility_citation(ctx: ToolContext, clinic: Clinic, *,
                       origin_lat: float, origin_lon: float, dist_mi: float,
                       valid_as_of: date | None = None) -> str:
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
        valid_as_of=valid_as_of or clinic.valid_as_of,
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

async def _query_hrsa(client: httpx.AsyncClient) -> _HrsaQueryResult:
    records: list[dict] = []
    seen_ids: set[str] = set()
    pages_fetched = 0
    offset = 0
    token = None
    while True:
        try:
            page = await query_feature_service_page(
                HRSA_URL,
                where=FQHC_WHERE,
                result_offset=offset,
                pagination_token=token,
                client=client,
            )
        except httpx.HTTPError:
            if not records:
                raise
            return _HrsaQueryResult(
                records=records,
                pages_fetched=pages_fetched,
                complete=False,
                next_offset=None if token else offset,
                pagination_token=token,
                error="transport_error",
            )
        except (ValueError, TypeError, AttributeError):
            if not records:
                raise
            return _HrsaQueryResult(
                records=records,
                pages_fetched=pages_fetched,
                complete=False,
                next_offset=None if token else offset,
                pagination_token=token,
                error="invalid_response",
            )
        page_ids = [str(record.get("OBJECTID") or "") for record in page.records]
        if (
            any(not record_id for record_id in page_ids)
            or len(set(page_ids)) != len(page_ids)
            or seen_ids.intersection(page_ids)
        ):
            if records:
                return _HrsaQueryResult(
                    records=records,
                    pages_fetched=pages_fetched,
                    complete=False,
                    next_offset=offset,
                    pagination_token=page.pagination_token or token,
                    error="invalid_response",
                )
            raise ValueError("HRSA page lacks stable unique OBJECTIDs")
        seen_ids.update(page_ids)
        records.extend(page.records)
        pages_fetched += 1
        if page.complete:
            return _HrsaQueryResult(
                records=records,
                pages_fetched=pages_fetched,
                complete=True,
                pagination_token=page.pagination_token or token,
            )
        if page.pagination_token is not None:
            if page.pagination_token == token:
                return _HrsaQueryResult(
                    records=records,
                    pages_fetched=pages_fetched,
                    complete=False,
                    pagination_token=token,
                    error="invalid_response",
                )
            token = page.pagination_token
            continue
        if page.next_offset is None:
            return _HrsaQueryResult(
                records=records,
                pages_fetched=pages_fetched,
                complete=False,
                pagination_token=token,
                error="invalid_response",
            )
        offset = page.next_offset

def _origin_result(point: GeoPoint, query: str) -> ClinicOrigin:
    return ClinicOrigin(
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


def _nyc_care_source(catalog: _NycCareCatalog) -> NycCareClinicSource:
    status: Literal["ok", "partial", "empty", "unavailable"]
    if catalog.error == "file_unavailable":
        status = "unavailable"
    elif not catalog.complete:
        status = "partial"
    elif not catalog.clinics:
        status = "empty"
    else:
        status = "ok"
    return NycCareClinicSource(
        status=status,
        returned_count=catalog.returned_count,
        usable_count=len(catalog.clinics),
        complete=catalog.complete,
        error=catalog.error,
    )


def _fallback_routes(ctx: ToolContext) -> list[ClinicRoute]:
    nyc_care_citation = _program_citation(ctx, CLASS_NYC_CARE)
    hrsa_url = "https://findahealthcenter.hrsa.gov/"
    hrsa_citation = ctx.citations.register(
        hrsa_url,
        snippet="Find a Health Center",
        title="Find a Health Center, HRSA",
        kind="DOC",
    )
    return [
        ClinicRoute(
            name="NYC Care",
            url=CLASS_GUARANTEE[CLASS_NYC_CARE].doc_url,
            phone="646-692-2273",
            citation_id=nyc_care_citation,
        ),
        ClinicRoute(
            name="HRSA Find a Health Center",
            url=hrsa_url,
            citation_id=hrsa_citation,
        ),
    ]


def _not_called_sources() -> ClinicSources:
    return ClinicSources(
        hrsa=HrsaClinicSource(status="not_called"),
        nyc_care=NycCareClinicSource(
            status="not_called",
            returned_count=0,
            usable_count=0,
            complete=False,
        ),
    )


def _source_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


async def _handler(args: dict, ctx: ToolContext) -> ClinicResult:
    query = ClinicQuery.model_validate(args)
    near = query.near.strip()
    if not near:
        return ClinicResult(outcome="missing_origin", sources=_not_called_sources())

    origin = await resolve_location(near, ctx)
    if origin is None:
        return ClinicResult(
            outcome="location_not_found",
            sources=_not_called_sources(),
            fallback_routes=_fallback_routes(ctx),
        )
    if origin.low_confidence:
        return ClinicResult(
            outcome="location_ambiguous",
            origin=_origin_result(origin, near),
            sources=_not_called_sources(),
            fallback_routes=_fallback_routes(ctx),
        )
    if origin.resident_query:
        ctx.current_location = origin

    fetched_at = datetime.now().astimezone()
    hrsa_result: _HrsaQueryResult | None = None
    try:
        hrsa_result = await _query_hrsa(ctx.http)
    except httpx.HTTPError:
        hrsa_source = HrsaClinicSource(
            status="unavailable",
            fetched_at=fetched_at,
            complete=False,
            error="transport_error",
        )
        fqhcs = []
    except (ValueError, TypeError, AttributeError):
        hrsa_source = HrsaClinicSource(
            status="unavailable",
            fetched_at=fetched_at,
            complete=False,
            error="invalid_response",
        )
        fqhcs = []
    else:
        parsed = [clinic for clinic in (_fqhc_from_record(row) for row in hrsa_result.records)
                  if clinic is not None]
        hrsa_source = HrsaClinicSource(
            status="ok" if hrsa_result.complete else "partial",
            fetched_at=fetched_at,
            returned_count=len(hrsa_result.records),
            usable_count=len(parsed),
            complete=hrsa_result.complete,
            pages_fetched=hrsa_result.pages_fetched,
            next_offset=hrsa_result.next_offset,
            pagination_token=hrsa_result.pagination_token,
            error=hrsa_result.error,
        )
        fqhcs = parsed if hrsa_result.complete else []

    catalog = _load_nyc_care_seed()
    nyc_care_source = _nyc_care_source(catalog)
    clinics = fqhcs + (catalog.clinics if catalog.complete else [])
    sources = ClinicSources(hrsa=hrsa_source, nyc_care=nyc_care_source)
    if not clinics:
        return ClinicResult(
            outcome="source_unavailable",
            origin=_origin_result(origin, near),
            origin_citation_id=resolved_location_citation(ctx, origin),
            sources=sources,
            fallback_routes=_fallback_routes(ctx),
        )

    max_results = query.max_results or 5
    ranked = rank_nearby(
        origin,
        clinics,
        key=lambda clinic: (
            clinic.name.strip().casefold(),
            round(clinic.lat, 5),
            round(clinic.lon, 5),
        ),
        limit=max_results,
    )

    typed_clinics: list[ClinicRecord] = []
    classes_present: list[str] = []
    for clinic, distance_m in ranked:
        dist_mi = miles(distance_m)
        valid_as_of = _source_date(clinic.valid_as_of) if clinic.valid_as_of else None
        cite = _facility_citation(
            ctx,
            clinic,
            origin_lat=origin.lat,
            origin_lon=origin.lon,
            dist_mi=dist_mi,
            valid_as_of=valid_as_of,
        )
        guarantee = CLASS_GUARANTEE[clinic.klass]
        typed_clinics.append(ClinicRecord(
            provider_record_id=clinic.record_id,
            organization=ClinicOrganization(name=clinic.name),
            service=ClinicService(
                program_class=clinic.klass,
                program_label=guarantee.label,
            ),
            location=ClinicLocation(
                address=clinic.address or None,
                borough=clinic.borough or None,
                latitude=clinic.lat,
                longitude=clinic.lon,
            ),
            service_at_location=ClinicServiceAtLocation(),
            phone=ClinicPhone(number=clinic.phone) if clinic.phone else None,
            website=clinic.url or None,
            distance_miles=dist_mi,
            valid_as_of=valid_as_of,
            citation_id=cite,
            action_url=maps_link(clinic.lat, clinic.lon),
        ))
        if clinic.klass not in classes_present:
            classes_present.append(clinic.klass)

    guarantees = []
    for klass in classes_present:
        guarantee = CLASS_GUARANTEE[klass]
        guarantees.append(ClinicProgramGuarantee(
            program_class=klass,
            label=guarantee.label,
            verified_fact=guarantee.body,
            valid_as_of=date.fromisoformat(VERIFIED_ON),
            citation_id=_program_citation(ctx, klass),
        ))

    return ClinicResult(
        outcome=(
            "success"
            if hrsa_source.status == "ok" and nyc_care_source.status == "ok"
            else "degraded"
        ),
        origin=_origin_result(origin, near),
        origin_citation_id=resolved_location_citation(ctx, origin),
        primary_citation_id=typed_clinics[0].citation_id,
        sources=sources,
        clinics=typed_clinics,
        program_guarantees=guarantees,
        fallback_routes=_fallback_routes(ctx) if hrsa_source.status != "ok" else [],
        requested_count=max_results,
    )


# --- get_health_coverage_guidance: official coverage facts, each cited to its source page ---
#
# `find_clinics` answers "where can I get seen"; this answers "what coverage can I get, and is it
# safe" for an uninsured or undocumented New Yorker. The facts are STATIC but official (a program's
# guarantee, a state coverage rule), so they live here as grounded _Fact
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
    valid_as_of: str = COVERAGE_VERIFIED_ON


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
              "care."),
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

EMERGENCY_MEDICAID_APPLICATION = _Fact(
    url="https://www.health.ny.gov/health_care/medicaid/how_do_i_apply.htm",
    title="How to Apply for NY Medicaid, New York State Department of Health",
    snippet=("NYC residents can call the HRA Medicaid Helpline at (888) 692-6116 for help "
             "applying for Medicaid"),
    body=("For help applying in NYC, call the HRA Medicaid Helpline at (888) 692-6116. "
          "Representatives can help make sure you apply in the correct place."),
    valid_as_of="2026-08-11",
)

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

def _resolve_coverage_topic(raw: str) -> str | None:
    """Accept only the canonical topic already constrained by the tool schema."""
    key = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in _COVERAGE else None


async def _coverage_handler(args: dict, ctx: ToolContext) -> str:
    topic = _resolve_coverage_topic(args.get("topic", ""))
    if topic is None:
        return ("I don't have grounded coverage guidance for that. Use get_health_coverage_guidance with "
                "topic = 'emergency_care' (ER screening and stabilizing care), 'nyc_care' (low/no-cost care at NYC Health + Hospitals, no immigration "
                "questions) or 'emergency_medicaid' (coverage for a medical emergency regardless of "
                "immigration status). To find a specific clinic use find_clinics; for anything else, "
                "point the user to 311 or 646-NYC-CARE (646-692-2273).")
    facts = (_COVERAGE[topic],)
    if topic == "emergency_medicaid":
        facts += (EMERGENCY_MEDICAID_APPLICATION,)
    if topic == "public_charge":
        facts += (PUBLIC_CHARGE_FINAL_RULE,)
    lines = [COVERAGE_INTRO]
    for fact in facts:
        cite = ctx.citations.register(
            fact.url,
            snippet=fact.snippet,
            title=fact.title,
            kind="DOC",
            valid_as_of=fact.valid_as_of,
            provenance={"snapshot": {"verified_fact": fact.body}},
        )
        lines.append(f"- {fact.body} {{cite:{cite}}}")
    if topic == "public_charge":
        lines.append(COVERAGE_CLOSING)
    lines.append(
        "Report ONLY these grounded facts with their {cite:Sn}. Do not add or change a phone number, "
        "a dollar figure, an eligibility rule, service type, application office, counselor, "
        "screening step, or hospital instruction that is not in a cited fact. Do not add urgency, "
        "timing, or an intake script beyond what a cited fact states. Keep each fact's citation "
        "only on claims from that fact; do not attach it to another fact."
    )
    return "\n".join(lines)


def get_tools() -> list[Tool]:
    return [
        Tool(
            name="find_clinics",
            description=(
                "Find the nearest NYC safety-net clinics that will see someone regardless of "
                "insurance or immigration status, grounded + cited. Merges two sources: live HRSA "
                "Federally Qualified Health Centers (community health centers, sliding fee scale) and "
                "NYC Health + Hospitals / NYC Care sites (low/no-cost, doesn't ask immigration "
                "status). Pass `near` = the user's NYC address, neighborhood, or ZIP; optional `max_results` "
                "(default 5). Returns each site's name, address, borough, phone, distance, and CLASS, "
                "plus a grounded eligibility guarantee cited to the program's official page. Use for "
                "'doctor without insurance', 'free clinic', 'I'm undocumented and sick'. NEVER guess a "
                "clinic: if geocoding fails or none are near, it abstains and routes to 311 / "
                "646-NYC-CARE. The eligibility/immigration-safety text comes only from the program "
                "citation, never invented per-site."
            ),
            parameters=ClinicQuery.model_json_schema(),
            handler=_handler,
            return_type=ClinicResult,
            open_world=True,  # hits the live HRSA ArcGIS service + geocoder (NYC Care seed is bundled)
        ),
        Tool(
            name="get_health_coverage_guidance",
            description=(
                "Return official, cited guidance for four high-stakes health coverage situations. "
                "Topics: `nyc_care` "
                "(low/no-cost care at NYC Health + Hospitals, sliding-scale fees from $0, doesn't ask "
                "immigration status, enroll at 646-NYC-CARE) and `emergency_medicaid` (Medicaid for a "
                "medical emergency, including emergency labor and delivery, regardless of immigration "
                "status), `emergency_care` (ER screening and stabilizing care regardless of ability "
                "to pay), and `public_charge` (MOIA public-charge guidance). Pass `topic` = one of "
                "those four values. find_clinics answers WHERE "
                "to go; this answers WHAT coverage / IS IT SAFE. It appends an ActionNYC routing line "
                "only for public-charge questions. For delivery payment, use `emergency_medicaid`. "
                "Do not also call `nyc_care` unless the resident separately asks about ongoing "
                "non-emergency care. Never state a coverage rule or a public-charge conclusion from "
                "your own knowledge; report only what it returns, cited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": list(_COVERAGE),
                        "description": "Health coverage situation to retrieve",
                    },
                },
                "required": ["topic"],
            },
            handler=_coverage_handler,
            open_world=False,  # static official facts baked in + cited; no network call
        ),
    ]
