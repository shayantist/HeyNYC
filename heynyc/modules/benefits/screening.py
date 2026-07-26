"""NYC Benefits Screening API client: cached token auth, a PII guard, and the screening
call. Kept separate from tools.py so the tool handler stays readable. PII never leaves here."""
from __future__ import annotations

import copy
import re
import time
from typing import Optional

import httpx
from jsonschema import Draft202012Validator

_TOKEN_TTL = 3300.0  # 55 min (the API token expires at 3600s)
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}  # base -> (token, expires_at)

RENTAL_TYPES = (
    "NYCHA", "MarketRate", "RentControlled", "RentRegulatedHotel", "Section213",
    "LimitedDividendDevelopment", "MitchellLama", "RedevelopmentCompany", "HDFC",
    "FamilyHome", "Condo",
)
HOUSEHOLD_MEMBER_TYPES = (
    "HeadOfHousehold", "Child", "FosterChild", "StepChild", "Grandchild", "Spouse",
    "Parent", "FosterParent", "StepParent", "Grandparent", "SisterBrother",
    "StepSisterStepBrother", "BoyfriendGirlfriend", "DomesticPartner", "Unrelated", "Other",
)
FREQUENCIES = ("Weekly", "Biweekly", "Monthly", "Semimonthly", "Yearly")
INCOME_TYPES = (
    "Wages", "SelfEmployment", "Unemployment", "CashAssistance", "ChildSupport",
    "DisabilityMedicaid", "SSI", "SSDependent", "SSDisability", "SSSurvivor",
    "SSRetirement", "NYSDisability", "Veteran", "Pension", "DeferredComp", "WorkersComp",
    "Alimony", "Boarder", "Gifts", "Rental", "Investment",
)
EXPENSE_TYPES = (
    "ChildCare", "ChildSupport", "DependentCare", "Rent", "Medical", "Heating", "Cooling",
    "Mortgage", "Utilities", "Telephone", "InsurancePremiums",
)
HOUSEHOLD_FIELDS = frozenset({
    "cashOnHand", "livingRentalType", "livingRenting", "livingOwner",
    "livingStayingWithFriend", "livingHotel", "livingShelter", "livingPreferNotToSay",
})
PERSON_FIELDS = frozenset({
    "age", "student", "studentFulltime", "pregnant", "unemployed",
    "unemployedWorkedLast18Months", "blind", "disabled", "veteran", "benefitsMedicaid",
    "benefitsMedicaidDisability", "householdMemberType", "livingOwnerOnDeed",
    "livingRentalOnLease", "incomes", "expenses",
})
MONEY_ITEM_FIELDS = frozenset({"amount", "frequency", "type"})
PROGRAM_CODE_PATTERN = r"^S2R\d{3}$"
BOOLEAN_FACT_DESCRIPTION = (
    "If the resident explicitly supplied this fact, include the field with true or false. "
    "Omit it only when the fact is unknown; never infer it."
)


def _money_item_schema(types: tuple[str, ...], max_length: int, max_whole_digits: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "amount": {
                "type": "string",
                "maxLength": max_length,
                "pattern": rf"^\d{{1,{max_whole_digits}}}(?:\.\d{{1,2}})?$",
            },
            "frequency": {"type": "string", "enum": list(FREQUENCIES)},
            "type": {"type": "string", "enum": list(types)},
        },
        "required": ["amount", "frequency", "type"],
    }


def request_schema() -> dict:
    """The one screening contract used by both model tools and runtime validation."""
    specific_housing_flags = (
        "livingRenting",
        "livingOwner",
        "livingStayingWithFriend",
        "livingHotel",
        "livingShelter",
    )
    household_flags = {
        name: {"type": "boolean", "description": BOOLEAN_FACT_DESCRIPTION}
        for name in (*specific_housing_flags, "livingPreferNotToSay")
    }
    household_flags["livingPreferNotToSay"]["description"] += (
        " True only when every specific housing flag is false or omitted."
    )
    person_flags = {
        name: {"type": "boolean", "description": BOOLEAN_FACT_DESCRIPTION} for name in (
            "student", "studentFulltime", "pregnant", "unemployed",
            "unemployedWorkedLast18Months", "blind", "disabled", "veteran",
            "benefitsMedicaid", "benefitsMedicaidDisability", "livingOwnerOnDeed",
            "livingRentalOnLease",
        )
    }
    household_properties = {
        "cashOnHand": {
            "type": "string", "maxLength": 10,
            "pattern": r"^\d{1,7}(?:\.\d{1,2})?$",
            "description": "Numeric USD amount encoded as a string, for example '500'.",
        },
        "livingRentalType": {"type": "string", "enum": list(RENTAL_TYPES)},
        **household_flags,
    }
    person_properties = {
        "age": {"type": "number", "minimum": 0, "maximum": 999},
        "householdMemberType": {"type": "string", "enum": list(HOUSEHOLD_MEMBER_TYPES)},
        "incomes": {
            "type": "array",
            "minItems": 1,
            "items": _money_item_schema(INCOME_TYPES, 15, 12),
            "description": (
                "Reported income items only. Omit when the resident reports no income or the "
                "amount is unknown; never add a zero placeholder."
            ),
        },
        "expenses": {"type": "array", "items": _money_item_schema(EXPENSE_TYPES, 9, 6)},
        **person_flags,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "household": {
                "type": "object",
                "additionalProperties": False,
                "description": "Household-level housing and cash-on-hand fields. PII-free.",
                "properties": household_properties,
                "allOf": [
                    {
                        "if": {
                            "properties": {"livingPreferNotToSay": {"const": True}},
                            "required": ["livingPreferNotToSay"],
                        },
                        "then": {
                            "properties": {
                                name: {"const": False}
                                for name in specific_housing_flags
                            }
                        },
                    }
                ],
            },
            "persons": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "contains": {
                    "type": "object",
                    "properties": {"householdMemberType": {"const": "HeadOfHousehold"}},
                    "required": ["householdMemberType"],
                },
                "minContains": 1,
                "description": "1-8 people; at least one must be HeadOfHousehold.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": person_properties,
                    "required": ["age", "householdMemberType"],
                },
            },
            "interested_programs": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string", "pattern": PROGRAM_CODE_PATTERN},
                "description": (
                    "Optional official program-code filter such as S2R007. Never pass a program "
                    "name such as SNAP; omit this field when the exact code is unknown."
                ),
            },
            "lang": {
                "type": "string",
                "description": "Optional language name, for example Spanish.",
            },
            "goal": {
                "type": "string", "maxLength": 200,
                "description": (
                    "Optional local-only need the resident explicitly stated, such as help buying "
                    "food. Used to shortlist results and never sent to the City API."
                ),
            },
            "show_all": {
                "type": "boolean",
                "description": "True only when the resident explicitly asks to see every match.",
            },
        },
        "required": ["persons"],
    }


def clear_token(base: str) -> None:
    _TOKEN_CACHE.pop(base, None)


async def get_token(client: httpx.AsyncClient, base: str, username: str, password: str) -> str:
    cached = _TOKEN_CACHE.get(base)
    now = time.time()
    if cached and cached[1] > now:
        return cached[0]
    resp = await client.post(f"{base}/authToken", json={"username": username, "password": password})
    resp.raise_for_status()
    token = (resp.json() or {}).get("token", "")
    _TOKEN_CACHE[base] = (token, now + _TOKEN_TTL)
    return token


_PII_KEYS = {"name", "firstname", "lastname", "fullname", "dob", "dateofbirth", "birthdate",
             "ssn", "social", "socialsecurity", "address", "street", "email", "phone"}
# Value-level identifier classes. These MIRROR the redaction patterns in
# heynyc/channels/analytics.py's redact_pii (SSN / DOB / phone / email / street / A-number /
# card) so the screener guard rejects exactly the classes the feedback log masks and the
# red-team suite probes. Ordered card/A-number first, like redact_pii, so the longest runs are
# classified before a phone/SSN pattern can nibble a fragment out of them.
_SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_ANUMBER_RE = re.compile(r"\bA[#\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2,3}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"\b\d(?:[\s-]?\d){14,18}\b")
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'#-]+\s+){0,3}"
    r"(?:street|st|avenue|ave|av|boulevard|blvd|road|rd|place|pl|drive|dr|lane|ln|court|ct|"
    r"parkway|pkwy|plaza|terrace|ter|way|broadway|highway|hwy)\b\.?",
    re.IGNORECASE,
)
_PII_VALUE_PATTERNS = (
    ("a card/EBT number", _CARD_RE),
    ("an immigration A-number", _ANUMBER_RE),
    ("an email address", _EMAIL_RE),
    ("an SSN", _SSN_RE),
    ("a phone number", _PHONE_RE),
    ("a date of birth", _DOB_RE),
    ("a street address", _STREET_RE),
)


def _pii_value_kind(value: str) -> Optional[str]:
    """Name the PII identifier class a string value looks like, or None if it is clean.
    Only strings are scanned: bare numbers (income amounts, household counts, ages, a zip) are
    legitimate profile values and must never be rejected, so numeric fields are left alone."""
    for label, pattern in _PII_VALUE_PATTERNS:
        if pattern.search(value):
            return label
    return None


def _assert_pii_free(node: object, path: str) -> None:
    """Walk a profile node recursively (dicts and lists), rejecting a PII key name at any depth
    and any string value that matches a PII identifier class. The path names where it slipped in."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.lower().replace("_", "") in _PII_KEYS:
                raise ValueError(f"PII field '{path}.{key}' is not allowed in a screening profile.")
            _assert_pii_free(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for i, item in enumerate(node):
            _assert_pii_free(item, f"{path}[{i}]")
    elif isinstance(node, str):
        kind = _pii_value_kind(node)
        if kind is not None:
            raise ValueError(f"a profile value at '{path}' looks like {kind}: PII must not be sent.")


def assert_pii_free(household: dict, persons: list[dict]) -> None:
    """Reject any PII before it can leave the box (the API forbids it AND it's our rule).
    The screener takes age / household-type / income only, never names, DOB, SSN, or address.
    Walks nested dicts and lists (persons[].incomes) so PII cannot hide below the top level, and
    scans string values for SSN / DOB / phone / email / street / A-number / EBT-card, the same
    classes redact_pii masks. Bare numbers stay legitimate and pass."""
    _assert_pii_free(household, "household")
    _assert_pii_free(persons, "persons")


def _canonical(value: object, allowed: tuple[str, ...]) -> object:
    """Restore the City's case-sensitive enum spelling without guessing unknown values."""
    if not isinstance(value, str):
        return value
    return {item.casefold(): item for item in allowed}.get(value.casefold(), value)


def validate_arguments(request: object) -> None:
    """Validate a complete tool payload against the exact model-facing schema."""
    error = next(iter(Draft202012Validator(request_schema()).iter_errors(request)), None)
    if error is None:
        return
    path = ".".join(str(part) for part in error.absolute_path) or "request"
    if error.validator == "contains":
        detail = "at least one person must have householdMemberType HeadOfHousehold"
    elif error.validator == "const" and "allOf" in error.absolute_schema_path:
        detail = (
            "livingPreferNotToSay cannot be true with a specific housing flag; "
            "omit it when the resident supplied a housing situation"
        )
    else:
        detail = error.message
    raise ValueError(f"invalid screening profile at {path}: {detail}")


def validate_request(household: dict, persons: list[dict], interested: Optional[list[str]] = None) -> None:
    """Validate the City request subset before any authentication or screening network call."""
    request = {"household": household, "persons": persons}
    if interested is not None:
        request["interested_programs"] = interested
    validate_arguments(request)


async def screen(client: httpx.AsyncClient, base: str, token: str,
                 household: dict, persons: list[dict],
                 interested: Optional[list[str]] = None) -> dict:
    validate_request(household, persons, interested)
    url = f"{base}/eligibilityPrograms"
    if interested:
        url += "?interestedPrograms=" + "|".join(interested)
    wire_household = copy.deepcopy(household)
    wire_persons = copy.deepcopy(persons)
    if "cashOnHand" in wire_household:
        wire_household["cashOnHand"] = str(wire_household["cashOnHand"])
    if "livingRentalType" in wire_household:
        wire_household["livingRentalType"] = _canonical(
            wire_household["livingRentalType"], RENTAL_TYPES
        )
    for person in wire_persons:
        if "householdMemberType" in person:
            person["householdMemberType"] = _canonical(
                person["householdMemberType"], HOUSEHOLD_MEMBER_TYPES
            )
        for item in person.get("incomes", []):
            if "amount" in item:
                item["amount"] = str(item["amount"])
            if "type" in item:
                item["type"] = _canonical(item["type"], INCOME_TYPES)
            if "frequency" in item:
                item["frequency"] = _canonical(item["frequency"], FREQUENCIES)
        for item in person.get("expenses", []):
            if "amount" in item:
                item["amount"] = str(item["amount"])
            if "type" in item:
                item["type"] = _canonical(item["type"], EXPENSE_TYPES)
            if "frequency" in item:
                item["frequency"] = _canonical(item["frequency"], FREQUENCIES)
    body = [{"household": [wire_household], "person": wire_persons, "withholdPayload": True}]
    resp = await client.post(url, json=body, headers={"Authorization": token})
    resp.raise_for_status()
    return resp.json() or {}


def request_summary(household: dict, persons: list[dict]) -> dict:
    """A REDACTED structural summary for the provenance snapshot, counts/flags, never amounts."""
    has_income = any(p.get("incomes") for p in persons)
    return {"persons": len(persons), "household_keys": sorted(household.keys()), "has_income": has_income}
