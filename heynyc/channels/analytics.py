"""Pseudonymous, operational analytics (keyed by user_key, never a phone number) +
a feedback log of user-flagged turns shaped to feed the agent-as-judge later."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from heynyc.core import telemetry
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
    with Path(path).open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# PII redaction for resident-authored free text (a flag note or the flagged query can carry the
# resident's own phone / SSN / email / home address). Patterns MIRROR the ones already proven in
# the codebase (core/grounding.py's phone regex, modules/benefits/screening.py's SSN/DOB regex) so
# the feedback log is PII-minimized by construction, same as sessions and telemetry. Over-masking a
# resident's own free text is the safe direction; the grounded agent answer is NOT run through this
# (its numbers are civic lines a reviewer needs to see).
_REDACTION = "[redacted]"
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
_DOB_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'#-]+\s+){0,3}"
    r"(?:street|st|avenue|ave|av|boulevard|blvd|road|rd|place|pl|drive|dr|lane|ln|court|ct|"
    r"parkway|pkwy|plaza|terrace|ter|way|broadway|highway|hwy)\b\.?",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """Mask phone / SSN / DOB / email / street-address spans in resident free text at write time."""
    if not text:
        return text or ""
    for pattern in (_EMAIL_RE, _SSN_RE, _PHONE_RE, _DOB_RE, _STREET_RE):
        text = pattern.sub(_REDACTION, text)
    return text


def record_feedback(
    path: Path, *, user_key: str, channel: str, message_id: str, flag: str,
    note: str, user_query: str, agent_text: str, ts: str | None = None,
) -> dict:
    """Build one PII-redacted feedback record for a user-flagged turn and append it (JSONL).

    Free text the resident authored (their `note` and the flagged `user_query`) is redacted at
    write time; the grounded `agent_text` is kept verbatim so the reviewer sees what was actually
    said. Keyed off the salted `user_key` (never a raw sender). Enough context to find and fix a
    systematic error without re-running the agent: the query, the answer, the reason, a timestamp."""
    record = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "user_key": user_key,
        "channel": channel,
        "message_id": message_id,
        "flag": flag,
        "note": redact_pii(note),
        "user_query": redact_pii(user_query),
        "agent_text": agent_text,
    }
    feedback_log(path, record)
    return record


def load_feedback(path: Path) -> list[dict]:
    """Read the append-only feedback log (JSONL) back into records."""
    return telemetry.load(Path(path))


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
