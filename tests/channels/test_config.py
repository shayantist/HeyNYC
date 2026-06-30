import importlib


def test_channel_config_defaults(monkeypatch):
    # Hermetic: ignore the developer's real .env (which may set these) so we test the
    # in-code defaults, then restore the real config for the rest of the suite.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    for var in ("WHATSAPP_PROVIDER", "HEYNYC_CHANNEL_RATE_LIMIT",
                "HEYNYC_CHANNEL_MAX_CONCURRENCY", "HEYNYC_PII_SALT"):
        monkeypatch.delenv(var, raising=False)
    from heynyc.core import config
    importlib.reload(config)
    try:
        assert config.WHATSAPP_PROVIDER == "meta"
        assert config.CHANNEL_RATE_LIMIT == 20
        assert config.CHANNEL_MAX_CONCURRENCY == 8
        assert config.HEYNYC_PII_SALT == ""
        assert hasattr(config, "TWILIO_WHATSAPP_FROM")
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # restore the real .env-backed config for other tests
