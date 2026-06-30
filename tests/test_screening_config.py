import importlib


def test_screening_creds_resolves_active_env(monkeypatch):
    monkeypatch.setenv("SCREENING_ENV", "sandbox")
    monkeypatch.setenv("SCREENING_SANDBOX_USERNAME", "u")
    monkeypatch.setenv("SCREENING_SANDBOX_PASSWORD", "p")
    from heynyc.core import config
    importlib.reload(config)
    base, user, pw = config.screening_creds()
    assert base == "https://sandbox.screeningapi.cityofnewyork.us"
    assert (user, pw) == ("u", "p")
    importlib.reload(config)  # restore
