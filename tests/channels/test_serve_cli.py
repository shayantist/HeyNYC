import sys

import pytest

pytest.importorskip("fastapi")


def test_serve_builds_app_and_runs_uvicorn(monkeypatch):
    calls = {}
    monkeypatch.setattr("heynyc.channels.app.create_app", lambda provider: ("APP", provider))

    def fake_run(app, host, port):
        calls.update(app=app, host=host, port=port)

    monkeypatch.setitem(sys.modules, "uvicorn", type("U", (), {"run": staticmethod(fake_run)}))
    monkeypatch.setattr(sys, "argv", ["heynyc", "serve", "--provider", "twilio", "--port", "9000"])
    from heynyc.__main__ import main
    main()
    assert calls["app"] == ("APP", "twilio")
    assert calls["port"] == 9000
