"""The PII boundary: a sender becomes a salted, non-reversible key. Raw phone
numbers never get persisted, sessions, telemetry, and feedback all key off this."""
from __future__ import annotations

import hmac


def user_key(channel: str, sender: str, salt: str) -> str:
    """Salted HMAC-SHA256 of `channel:sender`, truncated to 16 hex chars."""
    digest = hmac.new(salt.encode(), f"{channel}:{sender}".encode(), "sha256").hexdigest()
    return digest[:16]
