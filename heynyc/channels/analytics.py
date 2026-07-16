"""Pseudonymous operational analytics and encrypted user-flagged feedback."""
from __future__ import annotations

import base64
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from heynyc.core import pii_crypto, telemetry
from heynyc.eval.trace import classify_outcome


def record_interaction(*, telemetry_path: Path, model: str, user_key: str, channel: str, result) -> dict:
    grounded = bool({c.get("kind") for c in result.citations.values()} & {"DATA", "DOC"})
    outcome = classify_outcome(
        result.text, "error" if result.status == "error" else "success", grounded=grounded
    )
    return telemetry.record_turn(
        telemetry_path, session_id=user_key, model=model, usage=result.usage,
        n_tool_calls=len(result.tool_calls_made), tool_names=result.tool_calls_made,
        status=result.status,
        extra={"channel": channel, "outcome": outcome, "n_citations": len(result.citations)},
    )


def feedback_log(path: Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record)
    if pii_crypto.is_enabled():
        line = base64.b64encode(pii_crypto.encrypt(line)).decode("ascii")
    with Path(path).open("a") as fh:
        fh.write(line + "\n")


def _decode_feedback_line(line: str) -> dict:
    if pii_crypto.is_enabled():
        return json.loads(pii_crypto.decrypt(base64.b64decode(line)))
    return json.loads(line)


def migrate_plaintext_feedback(path: Path) -> bool:
    """Encrypt a legacy cleartext feedback log before hosted traffic is accepted."""
    path = Path(path)
    if not pii_crypto.is_enabled() or not path.exists():
        return False
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    migrated = False
    encoded: list[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            _decode_feedback_line(line)
            encoded.append(line)
        else:
            encoded.append(base64.b64encode(pii_crypto.encrypt(json.dumps(record))).decode("ascii"))
            migrated = True
    if migrated:
        replacement = path.with_suffix(path.suffix + ".tmp")
        replacement.write_text("\n".join(encoded) + ("\n" if encoded else ""))
        replacement.replace(path)
    return migrated


# PII redaction for resident-authored free text (a flag note or the flagged query can carry the
# resident's own phone / SSN / email / home address). Patterns MIRROR the ones already proven in
# the codebase (core/grounding.py's phone regex, modules/benefits/screening.py's SSN/DOB regex) so
# the feedback log is PII-minimized by construction, same as sessions and telemetry. Over-masking a
# resident's own free text is the safe direction. Assistant text is also redacted because it can
# echo a resident's message.
_REDACTION = "[redacted]"
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
_DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
# An immigration A-number / USCIS number (a capital A then 8-9 digits, any grouping) and a
# benefit/debit card number (NYC EBT is 19 digits; bank cards 15-16): the two identifiers a
# mixed-status or benefits household is most exposed by, which the phone/SSN patterns above miss.
# HeyNYC's own red-team suite probes both (redteam_suite.yaml A-number + EBT cases); mask them here
# so a resident who pastes one into a flag note or query never persists it in the clear.
_ANUMBER_RE = re.compile(r"\bA[#\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2,3}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"\b\d(?:[\s-]?\d){14,18}\b")
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'#-]+\s+){0,3}"
    r"(?:street|st|avenue|ave|av|boulevard|blvd|road|rd|place|pl|drive|dr|lane|ln|court|ct|"
    r"parkway|pkwy|plaza|terrace|ter|way|broadway|highway|hwy)\b\.?",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """Mask phone / SSN / DOB / email / street-address / A-number / card-number spans in resident
    free text at write time. Long card runs and A-numbers are masked first so a phone/SSN pattern
    can't nibble a fragment out of them."""
    if not text:
        return text or ""
    for pattern in (_CARD_RE, _ANUMBER_RE, _EMAIL_RE, _SSN_RE, _PHONE_RE, _DOB_RE, _STREET_RE):
        text = pattern.sub(_REDACTION, text)
    return text


def record_feedback(
    path: Path, *, user_key: str, channel: str, message_id: str, flag: str,
    note: str, user_query: str, agent_text: str, ts: str | None = None,
) -> dict:
    """Build one PII-redacted feedback record for a user-flagged turn and append it (JSONL).

    Free text in the note, query, and assistant answer is redacted at write time. Keyed off the
    salted `user_key` (never a raw sender). Enough context to find and fix a
    systematic error without re-running the agent: the query, the answer, the reason, a timestamp."""
    record = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "user_key": user_key,
        "channel": channel,
        "message_id": message_id,
        "flag": flag,
        "note": redact_pii(note),
        "user_query": redact_pii(user_query),
        "agent_text": redact_pii(agent_text),
    }
    feedback_log(path, record)
    return record


def load_feedback(path: Path) -> list[dict]:
    """Read the append-only feedback log (JSONL) back into records."""
    path = Path(path)
    if not path.exists():
        return []
    return [_decode_feedback_line(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_feedback(records: list[dict]) -> dict:
    """Aggregate flagged turns for the owner's review report: totals, flag/channel mix, and the
    repeat-flagged queries that point at a systematic error, plus the most recent flags."""
    if not records:
        return {"total": 0, "users": 0, "by_flag": {}, "by_channel": {},
                "top_queries": [], "recent": []}
    by_query = Counter(
        (r.get("user_query") or "").strip() for r in records if (r.get("user_query") or "").strip()
    )
    recent = sorted(records, key=lambda r: r.get("ts", ""), reverse=True)
    return {
        "total": len(records),
        "users": len({r.get("user_key") for r in records}),
        "by_flag": dict(Counter(r.get("flag", "") for r in records)),
        "by_channel": dict(Counter(r.get("channel", "") for r in records)),
        "top_queries": by_query.most_common(10),
        "recent": recent[:10],
    }
