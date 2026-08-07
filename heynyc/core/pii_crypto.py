"""At-rest encryption + retention policy for apply-time PII (security-audit F1).

The apply layer is the only path that touches real PII (name / DOB / SSN /
address typed to fill a SNAP draft). This module is its data-at-rest guard.

Design references (named on purpose, per AGENTS.md):
  - NIST SP 800-111 (guide to storage encryption for end-user data at rest).
  - AES-256-GCM, a FIPS-approved AEAD (NIST SP 800-38D): confidentiality AND
    integrity in one pass, so a tampered file fails to decrypt.
  - GDPR Art 5(1)(c)/(e): data minimization + storage limitation, which the
    retention sweep (irreversible delete of stale files) enforces.

Crypto choice, verified against the pyca/cryptography docs (AEAD recipe,
docs/hazmat/primitives/aead.rst): use the high-level ``AESGCM`` recipe with a
256-bit key and a FRESH RANDOM 96-bit nonce per ``encrypt`` call. The recipe
appends the 16-byte authentication tag to the ciphertext; we prepend the nonce,
so a token is ``nonce(12) || ciphertext || tag``. A nonce is NEVER reused with a
key (each call draws a new ``os.urandom(12)``). Decrypt raises
``cryptography.exceptions.InvalidTag`` on any tamper / wrong key / wrong nonce.

Key management (mirrors HEYNYC_PII_SALT): a base64-encoded 32-byte key in the
``HEYNYC_PII_KEY`` env var, read via ``os.getenv`` at call time. Generate one
with ``generate_key()`` and put it in ``.env`` next to ``HEYNYC_PII_SALT`` (never
commit it).

Fail-closed posture:
  - No key set  -> encryption is DISABLED and callers keep today's CLEARTEXT
    behavior. This is the dev/test path and it is insecure BY DESIGN; a hosted
    deployment MUST set the key.
  - Key set but MALFORMED (bad base64 / wrong length) -> we RAISE
    ``PiiCryptoError`` rather than silently writing cleartext. ``is_enabled()``
    reports True for any non-empty key, so a store attempts to encrypt and fails
    LOUDLY instead of falling back to a cleartext write.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENV_KEY = "HEYNYC_PII_KEY"
_ENV_RETENTION_DAYS = "HEYNYC_PII_RETENTION_DAYS"

KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12  # 96-bit nonce, NIST SP 800-38D recommended length for GCM
DEFAULT_RETENTION_DAYS = 30.0


class PiiCryptoError(RuntimeError):
    """Encryption was expected but the key is absent/malformed, or a ciphertext
    failed to authenticate. Raised so callers FAIL CLOSED instead of persisting
    or trusting cleartext PII."""


def generate_key() -> str:
    """A fresh base64-encoded 32-byte (AES-256) key for HEYNYC_PII_KEY.

    Put the printed value in ``.env`` alongside ``HEYNYC_PII_SALT`` and never
    commit it. One-liner for an operator:

        python -c "from heynyc.core.pii_crypto import generate_key; print(generate_key())"
    """
    return base64.b64encode(AESGCM.generate_key(bit_length=KEY_BYTES * 8)).decode("ascii")


def is_enabled() -> bool:
    """True when a key is configured (any non-empty ``HEYNYC_PII_KEY``).

    When False, callers keep today's cleartext behavior (the dev/test path).
    A set-but-malformed key still reads as enabled on purpose: the store then
    tries to encrypt and fails closed rather than silently writing cleartext.
    """
    return bool(os.getenv(_ENV_KEY, "").strip())


def _load_key() -> bytes:
    raw = os.getenv(_ENV_KEY, "").strip()
    if not raw:
        raise PiiCryptoError(
            f"{_ENV_KEY} is not set; refusing to encrypt/decrypt (a write would persist "
            f"cleartext PII). Generate a key with heynyc.core.pii_crypto.generate_key()."
        )
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError as exc:  # binascii.Error (bad base64) subclasses ValueError
        raise PiiCryptoError(f"{_ENV_KEY} is not valid base64") from exc
    if len(key) != KEY_BYTES:
        raise PiiCryptoError(
            f"{_ENV_KEY} must decode to {KEY_BYTES} bytes (AES-256); got {len(key)}"
        )
    return key


def encrypt(plaintext: str, *, associated_data: bytes | None = None) -> bytes:
    """AES-256-GCM encrypt ``plaintext`` -> ``nonce(12) || ciphertext || tag``.

    A fresh random 96-bit nonce is drawn per call (never reused). Optional
    associated data authenticates unencrypted row identity. Fails closed if
    the key is absent or malformed.
    """
    key = _load_key()
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce,
        plaintext.encode("utf-8"),
        associated_data,
    )
    return nonce + ciphertext


def decrypt(token: bytes, *, associated_data: bytes | None = None) -> str:
    """Reverse of :func:`encrypt`.

    Raises ``PiiCryptoError`` if the key is absent/malformed, the token is too
    short to carry a nonce, or authentication fails (tampering / wrong key /
    wrong nonce / wrong associated data -> the library's ``InvalidTag``).
    """
    key = _load_key()
    if len(token) <= NONCE_BYTES:
        raise PiiCryptoError("ciphertext token is too short to contain a nonce")
    nonce, ciphertext = token[:NONCE_BYTES], token[NONCE_BYTES:]
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, associated_data)
    except InvalidTag as exc:
        raise PiiCryptoError(
            "PII ciphertext failed authentication (tampered, or wrong key)"
        ) from exc
    return plaintext.decode("utf-8")


def retention_days() -> float:
    """The configured at-rest retention window in days (``HEYNYC_PII_RETENTION_DAYS``,
    default 30). Garbage or non-positive values fall back to the default."""
    raw = os.getenv(_ENV_RETENTION_DAYS, "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        days = float(raw)
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return days if days > 0 else DEFAULT_RETENTION_DAYS


def purge_expired_files(
    directory: Path, pattern: str, max_age_days: float | None = None
) -> list[str]:
    """Irreversibly delete files in ``directory`` matching ``pattern`` whose mtime
    is older than the retention window. Returns the deleted paths (as strings).

    No soft-delete: this is a hard ``unlink``, the storage-limitation backstop
    for PII that a completion ``clear()`` did not remove. A missing directory is
    a no-op (returns ``[]``).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    age = retention_days() if max_age_days is None else max_age_days
    cutoff = time.time() - age * 86400
    deleted: list[str] = []
    for path in directory.glob(pattern):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                deleted.append(str(path))
        except OSError:
            continue  # a racing delete / permission issue: skip, do not crash the sweep
    return deleted
