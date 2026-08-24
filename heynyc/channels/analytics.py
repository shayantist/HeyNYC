"""Pseudonymous operational analytics and encrypted user-flagged feedback."""
from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from heynyc.core import pii_crypto, telemetry
from heynyc.core.citations import used_citations
from heynyc.core.pii_redaction import redact_pii
from heynyc.eval.trace import classify_outcome


def record_interaction(*, telemetry_path: Path, model: str, user_key: str, channel: str, result) -> dict:
    grounded = bool({c.get("kind") for c in result.citations.values()} & {"DATA", "DOC"})
    outcome = classify_outcome(
        result.text, "error" if result.status == "error" else "success", grounded=grounded
    )
    used = used_citations(result.text, result.citations)
    used_doc_citations = sum(citation.get("kind") == "DOC" for citation in used.values())
    diagnostics = result.diagnostics or {}
    rejections = diagnostics.get("validation_rejections") or []
    extra = {
        "channel": channel,
        "outcome": outcome,
        "n_citations": len(used),
        "used_doc_citations": used_doc_citations,
    }
    if result.status not in ("success", None):
        extra.update({
            "failure_type": diagnostics.get("failure_type", "unknown"),
            "validation_rejection_stages": list(dict.fromkeys(
                rejection["stage"] for rejection in rejections if rejection.get("stage")
            )),
            "validation_rejection_count": len(rejections),
            "retry_kinds": list(result.usage.get("retry_kinds") or []),
            "stalled_model_requests": int(
                result.usage.get("stalled_model_requests", 0) or 0
            ),
        })
    return telemetry.record_turn(
        telemetry_path, session_id=user_key, model=model, usage=result.usage,
        n_tool_calls=len(result.tool_calls_made), tool_names=result.tool_calls_made,
        status=result.status,
        extra=extra,
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
