"""NYC Benefits Screening API client: cached token auth, a PII guard, and the screening
call. Kept separate from tools.py so the tool handler stays readable. PII never leaves here."""
from __future__ import annotations

import re
import time
from typing import Optional

import httpx

_TOKEN_TTL = 3300.0  # 55 min (the API token expires at 3600s)
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}  # base -> (token, expires_at)


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
_SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def assert_pii_free(household: dict, persons: list[dict]) -> None:
    """Reject any PII before it can leave the box (the API forbids it AND it's our rule).
    The screener takes age / household-type / income only — never names, DOB, SSN, or address."""
    for blob in [household, *persons]:
        for key, value in blob.items():
            if key.lower().replace("_", "") in _PII_KEYS:
                raise ValueError(f"PII field '{key}' is not allowed in a screening profile.")
            if isinstance(value, str) and (_SSN_RE.search(value) or _DOB_RE.search(value)):
                raise ValueError("a profile value looks like an SSN/DOB — PII must not be sent.")


async def screen(client: httpx.AsyncClient, base: str, token: str,
                 household: dict, persons: list[dict],
                 interested: Optional[list[str]] = None) -> dict:
    url = f"{base}/eligibilityPrograms"
    if interested:
        url += "?interestedPrograms=" + "|".join(interested)
    body = [{"household": [household], "person": persons, "withholdPayload": True}]
    resp = await client.post(url, json=body, headers={"Authorization": token})
    resp.raise_for_status()
    return resp.json() or {}


def request_summary(household: dict, persons: list[dict]) -> dict:
    """A REDACTED structural summary for the provenance snapshot — counts/flags, never amounts."""
    has_income = any(p.get("incomes") for p in persons)
    return {"persons": len(persons), "household_keys": sorted(household.keys()), "has_income": has_income}
