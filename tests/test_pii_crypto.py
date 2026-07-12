"""AES-256-GCM at-rest encryption for apply-time PII (security-audit F1).

Crypto correctness is non-negotiable: round-trip, a fresh nonce per call,
authenticated (tamper/wrong-key fail), and fail-closed when the key is
absent-where-expected or malformed (never a silent cleartext write).
"""
from __future__ import annotations

import base64

import pytest

from heynyc.core import pii_crypto
from heynyc.core.pii_crypto import PiiCryptoError

_KEY = "HEYNYC_PII_KEY"


def _set_key(monkeypatch) -> str:
    key = pii_crypto.generate_key()
    monkeypatch.setenv(_KEY, key)
    return key


def test_round_trip(monkeypatch):
    _set_key(monkeypatch)
    secret = "Ana Diaz, SSN 123-45-6789, 123 Main St"
    assert pii_crypto.decrypt(pii_crypto.encrypt(secret)) == secret


def test_unicode_round_trip(monkeypatch):
    _set_key(monkeypatch)
    secret = "José Muñoz — 你好 — 🗽"  # noqa: RUF001 (test data, no em dash in prose)
    assert pii_crypto.decrypt(pii_crypto.encrypt(secret)) == secret


def test_fresh_nonce_each_call(monkeypatch):
    _set_key(monkeypatch)
    text = "same plaintext"
    a = pii_crypto.encrypt(text)
    b = pii_crypto.encrypt(text)
    assert a != b  # a fresh random nonce => distinct ciphertexts
    assert a[: pii_crypto.NONCE_BYTES] != b[: pii_crypto.NONCE_BYTES]  # distinct nonces
    assert pii_crypto.decrypt(a) == pii_crypto.decrypt(b) == text


def test_token_layout_is_nonce_prefixed(monkeypatch):
    _set_key(monkeypatch)
    token = pii_crypto.encrypt("x")
    # nonce(12) + ciphertext + 16-byte GCM tag => strictly longer than nonce+tag
    assert len(token) > pii_crypto.NONCE_BYTES + 16


def test_tamper_is_detected(monkeypatch):
    _set_key(monkeypatch)
    token = bytearray(pii_crypto.encrypt("do not tamper"))
    token[-1] ^= 0x01  # flip a ciphertext/tag bit
    with pytest.raises(PiiCryptoError):
        pii_crypto.decrypt(bytes(token))


def test_wrong_key_fails(monkeypatch):
    _set_key(monkeypatch)
    token = pii_crypto.encrypt("secret")
    monkeypatch.setenv(_KEY, pii_crypto.generate_key())  # rotate to a different key
    with pytest.raises(PiiCryptoError):
        pii_crypto.decrypt(token)


def test_truncated_token_fails(monkeypatch):
    _set_key(monkeypatch)
    with pytest.raises(PiiCryptoError):
        pii_crypto.decrypt(b"short")


def test_fail_closed_when_key_absent(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert pii_crypto.is_enabled() is False
    with pytest.raises(PiiCryptoError):
        pii_crypto.encrypt("would-be-cleartext")


def test_fail_closed_on_bad_base64(monkeypatch):
    monkeypatch.setenv(_KEY, "!!!not-base64!!!")
    # A set-but-malformed key still reads as ENABLED, so the store attempts to
    # encrypt and fails loudly rather than silently writing cleartext.
    assert pii_crypto.is_enabled() is True
    with pytest.raises(PiiCryptoError):
        pii_crypto.encrypt("x")


def test_fail_closed_on_wrong_length_key(monkeypatch):
    monkeypatch.setenv(_KEY, base64.b64encode(b"\x00" * 16).decode())  # 128-bit, not 256
    assert pii_crypto.is_enabled() is True
    with pytest.raises(PiiCryptoError):
        pii_crypto.encrypt("x")


def test_generate_key_is_32_bytes(monkeypatch):
    key = pii_crypto.generate_key()
    assert len(base64.b64decode(key)) == pii_crypto.KEY_BYTES  # AES-256


def test_is_enabled_tracks_env(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert pii_crypto.is_enabled() is False
    monkeypatch.setenv(_KEY, pii_crypto.generate_key())
    assert pii_crypto.is_enabled() is True
    monkeypatch.setenv(_KEY, "   ")  # whitespace-only counts as unset
    assert pii_crypto.is_enabled() is False
