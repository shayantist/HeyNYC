"""NYC Benefits Screening API client: cached token auth, a PII guard, and the screening
call. Kept separate from tools.py so the tool handler stays readable. PII never leaves here."""
from __future__ import annotations

import copy
import re
import time
from typing import Optional

import httpx

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
_PROGRAM_CODE_RE = re.compile(PROGRAM_CODE_PATTERN)


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


def _assert_known_fields(household: dict, persons: list[dict]) -> None:
    unknown = set(household) - HOUSEHOLD_FIELDS
    if unknown:
        raise ValueError(f"unsupported screening field(s): household.{', household.'.join(sorted(unknown))}")
    for index, person in enumerate(persons):
        unknown = set(person) - PERSON_FIELDS
        if unknown:
            prefix = f"persons[{index}]."
            raise ValueError(f"unsupported screening field(s): {prefix}{(', ' + prefix).join(sorted(unknown))}")
        for collection in ("incomes", "expenses"):
            for item_index, item in enumerate(person.get(collection, [])):
                unknown = set(item) - MONEY_ITEM_FIELDS
                if unknown:
                    prefix = f"persons[{index}].{collection}[{item_index}]."
                    raise ValueError(
                        f"unsupported screening field(s): {prefix}{(', ' + prefix).join(sorted(unknown))}"
                    )
    if not any(
        isinstance(person.get("householdMemberType"), str)
        and person["householdMemberType"].casefold() == "headofhousehold".casefold()
        for person in persons
    ):
        raise ValueError("at least one person must have householdMemberType HeadOfHousehold")


def validate_request(household: dict, persons: list[dict], interested: Optional[list[str]] = None) -> None:
    """Validate the local City request contract before authentication or screening network calls."""
    _assert_known_fields(household, persons)
    if interested and any(
        not isinstance(code, str) or _PROGRAM_CODE_RE.fullmatch(code) is None
        for code in interested
    ):
        raise ValueError(
            "interested_programs must use official codes like S2R007, not program names; "
            "omit the filter when the code is unknown"
        )


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
