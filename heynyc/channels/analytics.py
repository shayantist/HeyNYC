"""Pseudonymous, operational analytics (keyed by user_key, never a phone number) +
a feedback log of user-flagged turns shaped to feed the agent-as-judge later."""
from __future__ import annotations

import json
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
