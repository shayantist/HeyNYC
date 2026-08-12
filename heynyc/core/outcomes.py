"""Outcomes / funnel instrumentation (OTI Gap 5): report how many residents reached
the APPLY step, not just how many answers we gave.

The vertical is find -> understand -> APPLY, so a resident's turn can pass through:

    turns -> screened -> eligible programs shown -> apply started -> filled form ready

Where each milestone comes from (least-invasive by design):
  - `turns`         : every telemetry record (`heynyc/core/telemetry.py`).
  - `screened`      : the `screen_access_nyc_eligibility` tool fired that turn (telemetry `tool_names`).
  - `apply_started` : the `prepare_snap_application` tool fired (telemetry `tool_names`).
These three ride entirely on the EXISTING per-turn telemetry, with no new hook.

Two milestones need the tool RESULT, which the telemetry record does not carry, so
they are recorded explicitly (PII-free, keyed by the salted `user_key` exactly like
telemetry) into a sidecar `outcomes.jsonl`:
  - `eligible_shown`: the screener returned >= 1 likely-eligible program.
  - `form_ready`    : a filled application (a PDF / handoff) was actually produced.

Privacy (spec 7 / AGENTS rule 7): a milestone record holds ONLY
{ts, user_key, eligible_shown, form_ready}. No phone number, no name, and no form
answer ever touches this log. The helpers read the AgentResult transiently to compute
two booleans and persist nothing but the booleans.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# The tools whose firing marks a funnel step (read straight from telemetry `tool_names`).
SCREEN_TOOL = "screen_access_nyc_eligibility"
APPLY_TOOL = "prepare_snap_application"

# The funnel stages in order. `turns` is the entry; each later stage drops off the prior.
STAGES = ["turns", "screened", "eligible_shown", "apply_started", "form_ready"]


def default_path(data_dir: Path) -> Path:
    """The milestone sidecar log, beside telemetry.jsonl in the app data dir."""
    return Path(data_dir) / "outcomes.jsonl"


def screener_eligible(result) -> bool:
    """True when the screening tool returned at least one likely-eligible program.

    Inferred from the citations the screen handler registers: each eligible program gets
    its own DATA citation whose snippet carries `program_code` (the verdict citation does
    not), so >= 1 such citation means >= 1 program was shown. Program names only, no PII.
    Requires the screen tool to have fired that turn so an unrelated mention can't trigger it."""
    fired = SCREEN_TOOL in (getattr(result, "tool_calls_made", None) or [])
    citations = getattr(result, "citations", None) or {}
    return fired and any(
        "program_code" in (c.get("snippet") or "") for c in citations.values()
    )


def milestones_from_result(result, *, produced_artifact: bool) -> dict:
    """The two outcome milestones telemetry can't see, from the AgentResult plus the real
    artifact the channel just saw. Returns only booleans, never any PII."""
    return {
        "eligible_shown": screener_eligible(result),
        "form_ready": bool(produced_artifact),
    }


def record_milestone(path: Path, *, user_key: str,
                     eligible_shown: bool, form_ready: bool) -> dict | None:
    """Append one PII-free milestone record, or return None (and write nothing) when there
    is no outcome to record.

    Only turns that reached a real outcome (an eligible list shown, or a form produced)
    are logged, so the sidecar stays tiny; `screened` / `apply_started` keep coming from
    telemetry for every path."""
    if not (eligible_shown or form_ready):
        return None
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_key": user_key,
        "eligible_shown": bool(eligible_shown),
        "form_ready": bool(form_ready),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def funnel(telemetry_records: list[dict], outcome_records: list[dict] | None = None) -> dict:
    """The find -> understand -> APPLY funnel with per-stage drop-off.

    `turns` / `screened` / `apply_started` come from telemetry `tool_names` (every path);
    `eligible_shown` / `form_ready` come from the outcomes sidecar (recorded where the tool
    RESULT is known). Counts are stage totals; drop-off is the fall from each stage to the
    next (`lost` and `rate`). The funnel is monotone by construction (a form-ready turn also
    fired the apply tool, so it also appears in telemetry), so the nested counts stay
    consistent across the two logs without a per-row join. A NEGATIVE `lost` means a stage
    grew rather than shrank (e.g. someone went straight to apply without a shown eligible
    list), which is surfaced, not hidden."""
    outcome_records = outcome_records or []
    counts = {
        "turns": len(telemetry_records),
        "screened": sum(1 for r in telemetry_records if SCREEN_TOOL in (r.get("tool_names") or [])),
        "eligible_shown": sum(1 for r in outcome_records if r.get("eligible_shown")),
        "apply_started": sum(1 for r in telemetry_records if APPLY_TOOL in (r.get("tool_names") or [])),
        "form_ready": sum(1 for r in outcome_records if r.get("form_ready")),
    }
    dropoff = {}
    for prev, cur in zip(STAGES, STAGES[1:]):
        lost = counts[prev] - counts[cur]
        dropoff[cur] = {"from": prev, "lost": lost,
                        "rate": (lost / counts[prev]) if counts[prev] else 0.0}
    return {"counts": counts, "dropoff": dropoff}
