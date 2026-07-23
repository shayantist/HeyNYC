#!/usr/bin/env python3
"""Detect recent Twilio inbound SIDs that never reached the local inbox."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def reconcile(client, database: Path, recipients: list[str], since: datetime) -> dict:
    """Compare provider-side inbound SIDs with the durable local inbox."""
    since = _utc(since)
    provider_sids = set()
    timestamp_missing = 0
    for recipient in dict.fromkeys(recipients):
        for message in client.messages.stream(
            to=recipient, date_sent_after=since.date()
        ):
            if getattr(message, "direction", None) != "inbound":
                continue
            sent_at = getattr(message, "date_sent", None) or getattr(
                message, "date_created", None
            )
            if sent_at is None:
                timestamp_missing += 1
                provider_sids.add(message.sid)
            elif _utc(sent_at) >= since:
                provider_sids.add(message.sid)
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
        local_sids = {row[0] for row in db.execute("SELECT message_id FROM inbox")}
    found = provider_sids & local_sids
    missing = sorted(provider_sids - local_sids)
    return {
        "checked_after": since.isoformat(),
        "provider_inbound_count": len(provider_sids),
        "matched_inbox_count": len(found),
        "missing_count": len(missing),
        "missing_sids": missing,
        "timestamp_missing_count": timestamp_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--to", action="append", required=True, dest="recipients")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error("--hours must be positive")
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        parser.error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required")
    from twilio.rest import Client

    report = reconcile(
        Client(account_sid, auth_token),
        args.database,
        args.recipients,
        datetime.now(UTC) - timedelta(hours=args.hours),
    )
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(2 if report["missing_count"] else 0)


if __name__ == "__main__":
    main()
