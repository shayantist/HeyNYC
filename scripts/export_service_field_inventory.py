#!/usr/bin/env python3
"""Generate the service-location source-field comparison.

This is developer documentation only. It imports the typed adapter records, validates every
declared public model path, and writes one deterministic Markdown report. It does not fetch source
data, change resident behavior, or create a database.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path
from typing import NamedTuple, get_args, get_origin

from pydantic import BaseModel

from heynyc.core import config
from heynyc.modules.childcare.tools import ChildCareProgram
from heynyc.modules.clinics.tools import ClinicRecord
from heynyc.modules.food_pantries.tools import FoodHelpRecord
from heynyc.modules.wic.tools import WicRecord

OUTPUT = config.PROJECT_ROOT / "docs" / "service-location-field-inventory.md"
REVIEWED_ON = "2026-08-18"
STATUSES = {"mapped", "source_absent", "not_applicable", "adapter_unmapped", "derived"}


class Field(NamedTuple):
    concept: str
    model_path: str
    raw_fields: tuple[str, ...]
    status: str
    empty_value: str
    invalid_value: str
    note: str = ""


class Source(NamedTuple):
    key: str
    module: str
    label: str
    record_model: type[BaseModel]
    adapter_link: str
    source_link: str
    fields: tuple[Field, ...]


def _field(
    concept: str,
    model_path: str = "",
    raw_fields: tuple[str, ...] = (),
    status: str = "mapped",
    empty_value: str = "row-null possible",
    invalid_value: str = "rejected or normalized",
    note: str = "",
) -> Field:
    return Field(
        concept, model_path, raw_fields, status, empty_value, invalid_value, note
    )


def inventory() -> tuple[Source, ...]:
    return (
        Source(
            "foodhelp",
            "FoodHelp",
            "NYC FoodHelp ArcGIS",
            FoodHelpRecord,
            "../heynyc/modules/food_pantries/tools.py",
            "https://finder.nyc.gov/foodhelp/",
            (
                _field(
                    "organization.name",
                    "organization.name",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                    note="The provider program name is modeled as the service and location name.",
                ),
                _field(
                    "service.name",
                    "service.name",
                    ("program",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "service.type",
                    "service.service_type",
                    ("program_type",),
                    empty_value="pantry fallback",
                    invalid_value="pantry fallback",
                ),
                _field(
                    "service.eligibility",
                    "service.eligibility_description",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service.languages",
                    "service.language",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.name",
                    "location.name",
                    ("program",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "location.address",
                    "location.physical_address",
                    ("distadd", "dist_location_info", "distboro", "distzip"),
                    invalid_value="normalized",
                ),
                _field(
                    "location.borough",
                    raw_fields=("distboro",),
                    status="adapter_unmapped",
                    empty_value="row-null possible",
                    invalid_value="unassessed",
                    note="Folded into the address but not exposed as its own typed field.",
                ),
                _field(
                    "location.coordinates",
                    "location.latitude",
                    ("lat", "lon"),
                    empty_value="record unusable",
                    invalid_value="record unusable",
                    note="Longitude is the sibling typed field.",
                ),
                _field(
                    "location.accessibility",
                    "location.accessibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.id",
                    "service_at_location.id",
                    ("GlobalID",),
                    empty_value="empty identifier",
                    invalid_value="normalized",
                ),
                _field(
                    "service_at_location.facility_type",
                    status="not_applicable",
                    empty_value="not applicable",
                    invalid_value="not applicable",
                ),
                _field(
                    "schedule.published_hours",
                    "schedule.listed_hours",
                    (
                        "fp_{day}_open{1..3}",
                        "fp_{day}_close{1..3}",
                        "sk_{day}_open{1..3}",
                        "sk_{day}_close{1..3}",
                        "fp_days_orig",
                        "sk_days_orig",
                        "fp_notes",
                        "sk_notes",
                    ),
                    empty_value="unknown schedule",
                    invalid_value="conflict preserved",
                ),
                _field(
                    "contact.phone",
                    "phone.number",
                    ("org_phone",),
                    invalid_value="normalized",
                ),
                _field(
                    "contact.website",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "source.record_id",
                    "service_at_location.id",
                    ("GlobalID",),
                    empty_value="empty identifier",
                    invalid_value="normalized",
                ),
                _field(
                    "source.valid_as_of",
                    "valid_as_of",
                    ("EditDate",),
                    empty_value="unknown freshness",
                    invalid_value="rejected",
                ),
                _field(
                    "evaluation.distance",
                    "distance_miles",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic calculation",
                ),
                _field(
                    "evaluation.open_now",
                    "schedule.status",
                    status="derived",
                    empty_value="unknown when schedule is missing",
                    invalid_value="conflict or unknown",
                ),
                _field(
                    "action.url",
                    "action_url",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic URL",
                ),
            ),
        ),
        Source(
            "wic",
            "WIC",
            "Health Data NY WIC sites",
            WicRecord,
            "../heynyc/modules/wic/tools.py",
            "https://health.data.ny.gov/resource/g4i5-r6zx.json",
            (
                _field(
                    "organization.name",
                    "organization.name",
                    ("agency_name",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "service.name",
                    "service.name",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="fixed WIC label",
                ),
                _field(
                    "service.type",
                    status="not_applicable",
                    empty_value="not applicable",
                    invalid_value="not applicable",
                ),
                _field(
                    "service.eligibility",
                    "service.eligibility_description",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service.languages",
                    "service.language",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.name",
                    "location.name",
                    ("agency_name",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "location.address",
                    "location.physical_address",
                    ("street_address", "street2", "city", "state", "zip"),
                    invalid_value="normalized",
                ),
                _field(
                    "location.borough",
                    "location.borough",
                    ("counties_boroughs_served",),
                    invalid_value="normalized",
                ),
                _field(
                    "location.coordinates",
                    "location.latitude",
                    ("location_1.latitude", "location_1.longitude"),
                    empty_value="record unusable",
                    invalid_value="record unusable",
                    note="Longitude is the sibling typed field.",
                ),
                _field(
                    "location.accessibility",
                    "location.accessibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.id",
                    "service_at_location.id",
                    (":id",),
                    empty_value="derived fallback",
                    invalid_value="normalized",
                ),
                _field(
                    "service_at_location.facility_type",
                    "service_at_location.site_type",
                    ("site_type",),
                    invalid_value="preserved text",
                ),
                _field(
                    "schedule.published_hours",
                    "hours",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "contact.phone",
                    "phone.number",
                    ("phone_number",),
                    invalid_value="normalized",
                ),
                _field(
                    "contact.website",
                    "website",
                    ("link_to_website.url",),
                    invalid_value="cleaned text",
                ),
                _field(
                    "source.record_id",
                    "service_at_location.id",
                    (":id",),
                    empty_value="derived fallback",
                    invalid_value="normalized",
                ),
                _field(
                    "source.valid_as_of",
                    "valid_as_of",
                    (":updated_at",),
                    empty_value="unknown freshness",
                    invalid_value="rejected",
                ),
                _field(
                    "evaluation.distance",
                    "distance_miles",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic calculation",
                ),
                _field(
                    "evaluation.open_now",
                    status="not_applicable",
                    empty_value="no published schedule",
                    invalid_value="not applicable",
                ),
                _field(
                    "action.url",
                    "action_url",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic route",
                ),
            ),
        ),
        Source(
            "childcare",
            "Child care",
            "NYC regulated child care",
            ChildCareProgram,
            "../heynyc/modules/childcare/tools.py",
            "https://data.cityofnewyork.us/resource/gy3q-4tzp.json",
            (
                _field(
                    "organization.name",
                    "organization.name",
                    ("program_name",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "service.name",
                    "service.program_type",
                    ("program_type",),
                    invalid_value="empty label",
                ),
                _field(
                    "service.type",
                    "service.program_type",
                    ("program_type",),
                    invalid_value="empty label",
                ),
                _field(
                    "service.eligibility",
                    "service.eligibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service.languages",
                    "service.language",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.name",
                    "organization.name",
                    ("program_name",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                    note="The public record uses the organization name as the facility label.",
                ),
                _field(
                    "location.address",
                    "location.address",
                    ("address", "borough", "zipcode"),
                    invalid_value="normalized",
                ),
                _field(
                    "location.borough",
                    "location.borough",
                    ("borough",),
                    invalid_value="normalized",
                ),
                _field(
                    "location.coordinates",
                    "location.latitude",
                    ("latitude", "longitude"),
                    empty_value="excluded before ranking",
                    invalid_value="record unusable",
                    note="Longitude is the sibling typed field.",
                ),
                _field(
                    "location.accessibility",
                    "service.accessibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.id",
                    "provider_record_id",
                    (":id",),
                    empty_value="empty identifier",
                    invalid_value="normalized",
                ),
                _field(
                    "service_at_location.facility_type",
                    "service_at_location.facility_type",
                    ("facility_type",),
                    invalid_value="empty label",
                ),
                _field(
                    "schedule.published_hours",
                    "service_at_location.schedule",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "contact.phone",
                    "phone.number",
                    ("phone",),
                    invalid_value="normalized",
                ),
                _field(
                    "contact.website",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "source.record_id",
                    "provider_record_id",
                    (":id",),
                    empty_value="empty identifier",
                    invalid_value="normalized",
                ),
                _field(
                    "source.valid_as_of",
                    "valid_as_of",
                    (":updated_at",),
                    empty_value="unknown freshness",
                    invalid_value="rejected",
                ),
                _field(
                    "evaluation.distance",
                    "distance_miles",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic calculation",
                ),
                _field(
                    "evaluation.open_now",
                    status="not_applicable",
                    empty_value="no published schedule",
                    invalid_value="not applicable",
                ),
                _field(
                    "action.url",
                    "action_url",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic directory route",
                ),
            ),
        ),
        Source(
            "clinics_hrsa",
            "Clinics",
            "HRSA health centers",
            ClinicRecord,
            "../heynyc/modules/clinics/tools.py",
            "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/PrimaryHealthCareFacilities_FS/MapServer/0",
            (
                _field(
                    "organization.name",
                    "organization.name",
                    ("SITE_NM",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "service.name",
                    "service.program_label",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="fixed program label",
                ),
                _field(
                    "service.type",
                    "service.program_class",
                    ("HCC_STATUS_DESC",),
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="query-filtered to Active",
                ),
                _field(
                    "service.eligibility",
                    "service.eligibility",
                    status="source_absent",
                    empty_value="location source has no field",
                    invalid_value="not applicable",
                    note="Separate official program guidance is kept outside the location record.",
                ),
                _field(
                    "service.languages",
                    "service.language",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.name",
                    status="not_applicable",
                    empty_value="organization name is the facility label",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.address",
                    "location.address",
                    ("SITE_ADDRESS", "SITE_CITY", "SITE_ZIP_CD"),
                    invalid_value="normalized",
                ),
                _field(
                    "location.borough",
                    "location.borough",
                    ("COUNTY_NM",),
                    invalid_value="normalized",
                ),
                _field(
                    "location.coordinates",
                    "location.latitude",
                    ("lat", "lon"),
                    empty_value="record unusable",
                    invalid_value="record unusable",
                    note="Longitude is the sibling typed field.",
                ),
                _field(
                    "location.accessibility",
                    "service.accessibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.id",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.facility_type",
                    status="not_applicable",
                    empty_value="service class carries FQHC",
                    invalid_value="not applicable",
                ),
                _field(
                    "schedule.published_hours",
                    "service_at_location.schedule",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "contact.phone",
                    "phone.number",
                    ("SITE_PHONE_NUM",),
                    invalid_value="normalized",
                ),
                _field(
                    "contact.website",
                    "website",
                    ("SITE_URL",),
                    invalid_value="cleaned text",
                ),
                _field(
                    "source.record_id",
                    "provider_record_id",
                    ("OBJECTID",),
                    empty_value="empty identifier",
                    invalid_value="normalized",
                ),
                _field(
                    "source.valid_as_of",
                    "valid_as_of",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "evaluation.distance",
                    "distance_miles",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic calculation",
                ),
                _field(
                    "evaluation.open_now",
                    status="not_applicable",
                    empty_value="no published schedule",
                    invalid_value="not applicable",
                ),
                _field(
                    "action.url",
                    "action_url",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic map URL",
                ),
            ),
        ),
        Source(
            "clinics_nyc_care",
            "Clinics",
            "NYC Care maintained seed",
            ClinicRecord,
            "../heynyc/modules/clinics/tools.py",
            "https://www.nyccare.nyc/locations/",
            (
                _field(
                    "organization.name",
                    "organization.name",
                    ("name",),
                    empty_value="fallback label",
                    invalid_value="fallback label",
                ),
                _field(
                    "service.name",
                    "service.program_label",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="fixed program label",
                ),
                _field(
                    "service.type",
                    "service.program_class",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="fixed NYC_CARE class",
                ),
                _field(
                    "service.eligibility",
                    "service.eligibility",
                    status="source_absent",
                    empty_value="location seed has no field",
                    invalid_value="not applicable",
                    note="Separate official program guidance is kept outside the location record.",
                ),
                _field(
                    "service.languages",
                    "service.language",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.name",
                    status="not_applicable",
                    empty_value="organization name is the facility label",
                    invalid_value="not applicable",
                ),
                _field(
                    "location.address",
                    "location.address",
                    ("street", "borough", "zip"),
                    invalid_value="normalized",
                ),
                _field(
                    "location.borough",
                    "location.borough",
                    ("borough",),
                    invalid_value="normalized",
                ),
                _field(
                    "location.coordinates",
                    "location.latitude",
                    ("lat", "lon"),
                    empty_value="record unusable",
                    invalid_value="record unusable",
                    note="Longitude is the sibling typed field.",
                ),
                _field(
                    "location.accessibility",
                    "service.accessibility",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.id",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "service_at_location.facility_type",
                    raw_fields=("hh_type",),
                    status="adapter_unmapped",
                    empty_value="row-null possible",
                    invalid_value="unassessed",
                    note="Present in the maintained seed but dropped before the public typed record.",
                ),
                _field(
                    "schedule.published_hours",
                    "service_at_location.schedule",
                    status="source_absent",
                    empty_value="always absent",
                    invalid_value="not applicable",
                ),
                _field(
                    "contact.phone",
                    "phone.number",
                    ("phone",),
                    invalid_value="normalized",
                ),
                _field(
                    "contact.website",
                    "website",
                    ("url",),
                    invalid_value="cleaned text",
                ),
                _field(
                    "source.record_id",
                    "provider_record_id",
                    ("name",),
                    status="derived",
                    empty_value="empty identifier",
                    invalid_value="normalized identifier",
                ),
                _field(
                    "source.valid_as_of",
                    "valid_as_of",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="maintainer verification date",
                    note="Uses the checked-in seed verification date, not a provider row timestamp.",
                ),
                _field(
                    "evaluation.distance",
                    "distance_miles",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic calculation",
                ),
                _field(
                    "evaluation.open_now",
                    status="not_applicable",
                    empty_value="no published schedule",
                    invalid_value="not applicable",
                ),
                _field(
                    "action.url",
                    "action_url",
                    status="derived",
                    empty_value="not applicable",
                    invalid_value="deterministic map URL",
                ),
            ),
        ),
    )


def _model_type(annotation: object) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin in (types.UnionType, list, tuple):
        candidates = get_args(annotation)
    else:
        candidates = (
            get_args(annotation) if candidates_are_union(annotation) else (annotation,)
        )
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def candidates_are_union(annotation: object) -> bool:
    return str(get_origin(annotation)) == "typing.Union"


def _path_exists(model: type[BaseModel], path: str) -> bool:
    current = model
    parts = path.split(".")
    for index, part in enumerate(parts):
        field = current.model_fields.get(part)
        if field is None:
            return False
        if index < len(parts) - 1:
            nested = _model_type(field.annotation)
            if nested is None:
                return False
            current = nested
    return True


def validate_inventory(sources: tuple[Source, ...]) -> list[str]:
    errors: list[str] = []
    expected_concepts = {field.concept for field in sources[0].fields}
    for source in sources:
        concepts = [field.concept for field in source.fields]
        if len(concepts) != len(set(concepts)):
            errors.append(f"{source.key}: duplicate concepts")
        if set(concepts) != expected_concepts:
            errors.append(f"{source.key}: concept set differs")
        for field in source.fields:
            if field.status not in STATUSES:
                errors.append(
                    f"{source.key}/{field.concept}: unknown status {field.status}"
                )
            if field.model_path and not _path_exists(
                source.record_model, field.model_path
            ):
                errors.append(
                    f"{source.key}/{field.concept}: missing model path {field.model_path}"
                )
            if field.status == "adapter_unmapped" and field.model_path:
                errors.append(
                    f"{source.key}/{field.concept}: unmapped field has a model path"
                )
    return errors


def _cell(field: Field) -> str:
    raw = ", ".join(f"`{name}`" for name in field.raw_fields) or "none"
    return f"**{field.status}**<br>{raw}"


def render(sources: tuple[Source, ...]) -> str:
    errors = validate_inventory(sources)
    if errors:
        raise ValueError("; ".join(errors))
    lines = [
        "# Service-location source field inventory",
        "",
        "<!-- Generated by scripts/export_service_field_inventory.py. Do not edit by hand. -->",
        "",
        f"Adapter review date: **{REVIEWED_ON}**.",
        "",
        "This report compares the source fields preserved by HeyNYC's typed FoodHelp, WIC, child-care, and clinic adapters. It is an interchange and coverage report, not a database, resident response, or HSDS conformance claim. Each provider heading links to the adapter and responsible source used for that lane.",
        "",
        "Status meanings: **mapped** means a provider value reaches the typed record; **source_absent** means the adapter's source contract has no usable field; **not_applicable** means the concept does not fit that source; **adapter_unmapped** means the source provides a value that the typed record drops; **derived** means HeyNYC computes or supplies the value outside the provider row.",
        "",
        "Empty and invalid handling are separate from coverage. A mapped field may still be row-null, rejected, normalized, or preserved as an explicit conflict.",
        "",
        "## Cross-source comparison",
        "",
        "| Common concept | " + " | ".join(source.label for source in sources) + " |",
        "| --- | " + " | ".join("---" for _ in sources) + " |",
    ]
    by_source = [
        {field.concept: field for field in source.fields} for source in sources
    ]
    for concept in (field.concept for field in sources[0].fields):
        lines.append(
            f"| `{concept}` | "
            + " | ".join(_cell(fields[concept]) for fields in by_source)
            + " |"
        )
    lines.extend(["", "## Provider details", ""])
    for source in sources:
        lines.extend(
            [
                f"### {source.label}",
                "",
                f"Resident module: **{source.module}**. [Adapter]({source.adapter_link}) · [Responsible source]({source.source_link})",
                "",
                "| Common concept | Typed model path | Provider field or pattern | Coverage | Empty value | Invalid value | Notes |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for field in source.fields:
            path = f"`{field.model_path}`" if field.model_path else "none"
            raw = ", ".join(f"`{name}`" for name in field.raw_fields) or "none"
            lines.append(
                f"| `{field.concept}` | {path} | {raw} | {field.status} | "
                f"{field.empty_value} | {field.invalid_value} | {field.note} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary for the next implementation step",
            "",
            "Only `schedule.published_hours` belongs to the canonical source profile. `evaluation.distance`, `evaluation.open_now`, requested-date status, next-opening calculations, route URLs, query metadata, paging, and citations remain query-time or provenance metadata. The first shared profile should therefore be extracted from the mapped human-service concepts above without forcing event or amenity records into this shape.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render(inventory())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            print(f"service field inventory drifted: run {Path(__file__).name}")
            return 1
        print("service field inventory is current")
        return 0
    OUTPUT.write_text(content)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
