import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_health_and_twilio_route_mounted(monkeypatch, tmp_path):
    monkeypatch.setattr("heynyc.core.config.HEYNYC_PII_SALT", "x")
    monkeypatch.setattr("heynyc.core.config.HEYNYC_DATA_DIR", tmp_path)
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
