import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from heynyc.core import pii_crypto


def test_health_and_twilio_route_mounted(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "x")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    from heynyc.channels import app as appmod
    api = appmod.create_app(provider="twilio")
    with TestClient(api) as client:
        assert client.get("/health").json() == {"status": "ok"}
        # the twilio webhook is mounted: a request without a valid signature is
        # rejected (403), not missing (404).
        assert client.post("/webhook/twilio", data={"Body": "hi"}).status_code == 403


def test_missing_salt_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
    from heynyc.channels import app as appmod
    with pytest.raises(RuntimeError):
        appmod.create_app(provider="twilio")


def test_missing_encryption_key_raises_when_serving(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "x")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
    monkeypatch.delenv("HEYNYC_PII_KEY", raising=False)
    from heynyc.channels import app as appmod
    with pytest.raises(RuntimeError, match="HEYNYC_PII_KEY"):
        appmod.create_app(provider="twilio")


def test_lifespan_runs_private_data_purge(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "x")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    calls = []
    from heynyc.channels import app as appmod
    monkeypatch.setattr(appmod, "purge_private_data", lambda data: calls.append(data))
    monkeypatch.setattr(appmod, "purge_channel_data", lambda store: calls.append(store))
    api = appmod.create_app(provider="twilio")
    with TestClient(api):
        pass
    assert calls[0] == tmp_path
    assert calls[1] is api.state.deps.store


def test_lifespan_migrates_legacy_private_data_before_purge(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "x")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
    monkeypatch.setenv("HEYNYC_PII_KEY", pii_crypto.generate_key())
    calls = []
    from heynyc.channels import app as appmod
    monkeypatch.setattr(appmod, "migrate_private_data", lambda data: calls.append(("migrate", data)))
    monkeypatch.setattr(appmod, "purge_private_data", lambda data: calls.append(("purge", data)))
    with TestClient(appmod.create_app(provider="twilio")):
        pass
    assert calls == [("migrate", tmp_path), ("purge", tmp_path)]
