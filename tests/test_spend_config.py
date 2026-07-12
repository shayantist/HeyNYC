"""HEYNYC_SPEND_CAP env parsing (security-audit F2b). Default OFF; 0 / blank / garbage -> disabled."""
from __future__ import annotations

import importlib


def test_spend_cap_defaults_to_none_when_unset(monkeypatch):
    from heynyc.core import config

    monkeypatch.delenv("HEYNYC_SPEND_CAP", raising=False)
    importlib.reload(config)
    assert config.HEYNYC_SPEND_CAP is None
    importlib.reload(config)  # restore


def test_spend_cap_parses_a_positive_dollar_ceiling(monkeypatch):
    from heynyc.core import config

    monkeypatch.setenv("HEYNYC_SPEND_CAP", "5.00")
    importlib.reload(config)
    assert config.HEYNYC_SPEND_CAP == 5.0
    monkeypatch.delenv("HEYNYC_SPEND_CAP", raising=False)
    importlib.reload(config)  # restore


def test_spend_cap_zero_and_garbage_are_disabled(monkeypatch):
    from heynyc.core import config

    for raw in ("0", "0.0", "-1", "", "   ", "notanumber"):
        monkeypatch.setenv("HEYNYC_SPEND_CAP", raw)
        importlib.reload(config)
        assert config.HEYNYC_SPEND_CAP is None, raw
    monkeypatch.delenv("HEYNYC_SPEND_CAP", raising=False)
    importlib.reload(config)  # restore
