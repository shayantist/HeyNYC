from __future__ import annotations

import importlib.util
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "reconcile_twilio.py"


def _module():
    assert SCRIPT.exists(), "Twilio reconciliation script has not been implemented"
    spec = importlib.util.spec_from_file_location("reconcile_twilio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Messages:
    def __init__(self, by_recipient):
        self.by_recipient = by_recipient
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.by_recipient[kwargs["to"]])


def test_reconcile_detects_only_recent_inbound_sids_missing_locally(tmp_path: Path) -> None:
    reconcile = _module()
    database = tmp_path / "channels.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE inbox (message_id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO inbox VALUES ('SMstored')")
    now = datetime(2026, 7, 22, 20, tzinfo=UTC)
    since = now - timedelta(hours=2)
    sms = "+18882120042"
    whatsapp = "whatsapp:+18882120042"
    messages = _Messages(
        {
            sms: [
                SimpleNamespace(sid="SMstored", direction="inbound", date_sent=now),
                SimpleNamespace(sid="SMmissing", direction="inbound", date_sent=now),
                SimpleNamespace(sid="SMoutbound", direction="outbound-api", date_sent=now),
            ],
            whatsapp: [
                SimpleNamespace(sid="SMold", direction="inbound", date_sent=since - timedelta(seconds=1)),
                SimpleNamespace(sid="SMwa", direction="inbound", date_sent=now),
            ],
        }
    )

    report = reconcile.reconcile(
        SimpleNamespace(messages=messages), database, [sms, whatsapp], since
    )

    assert report == {
        "checked_after": since.isoformat(),
        "provider_inbound_count": 3,
        "matched_inbox_count": 1,
        "missing_count": 2,
        "missing_sids": ["SMmissing", "SMwa"],
        "timestamp_missing_count": 0,
    }
    assert messages.calls == [
        {"to": sms, "date_sent_after": since.date()},
        {"to": whatsapp, "date_sent_after": since.date()},
    ]


def test_reconcile_reports_a_complete_match_without_reading_message_content(tmp_path: Path) -> None:
    reconcile = _module()
    database = tmp_path / "channels.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE inbox (message_id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO inbox VALUES ('SMstored')")
    now = datetime(2026, 7, 22, 20, tzinfo=UTC)
    messages = _Messages(
        {
            "+18882120042": [
                SimpleNamespace(
                    sid="SMstored", direction="inbound", date_sent=now, body="resident text"
                ),
                SimpleNamespace(
                    sid="SMmissingtime", direction="inbound", date_sent=None, date_created=None
                ),
            ]
        }
    )

    report = reconcile.reconcile(
        SimpleNamespace(messages=messages),
        database,
        ["+18882120042"],
        now - timedelta(hours=1),
    )

    assert report["missing_count"] == 1
    assert report["timestamp_missing_count"] == 1
    assert "resident text" not in repr(report)
