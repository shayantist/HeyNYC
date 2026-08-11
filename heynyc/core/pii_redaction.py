"""Shared deterministic redaction for resident-authored free text."""
from __future__ import annotations

import re

_REDACTION = "[redacted]"
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
_DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_ANUMBER_RE = re.compile(r"\bA[#\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2,3}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"\b\d(?:[\s-]?\d){14,18}\b")
_ACCOUNT_RE = re.compile(r"\b\d{10,14}\b")
_PASSPORT_RE = re.compile(r"\b[A-Z]\d{7,8}\b", re.IGNORECASE)
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'#-]+\s+){0,3}"
    r"(?:street|st|avenue|ave|av|boulevard|blvd|road|rd|place|pl|drive|dr|lane|ln|court|ct|"
    r"parkway|pkwy|plaza|terrace|ter|way|broadway|highway|hwy)\b\.?",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """Mask common identifiers and street addresses in resident-authored text."""
    if not text:
        return text or ""
    for pattern in (
        _CARD_RE, _ACCOUNT_RE, _ANUMBER_RE, _PASSPORT_RE, _EMAIL_RE,
        _SSN_RE, _PHONE_RE, _DOB_RE, _STREET_RE,
    ):
        text = pattern.sub(_REDACTION, text)
    return text


def redact_sensitive_identifiers(text: str) -> str:
    """Mask identifiers that should never be needed for conversational guidance.

    Phone numbers and street addresses remain available for location and callback questions.
    """
    if not text:
        return text or ""
    for pattern in (_CARD_RE, _ACCOUNT_RE, _ANUMBER_RE, _PASSPORT_RE, _SSN_RE):
        text = pattern.sub(_REDACTION, text)
    return text
