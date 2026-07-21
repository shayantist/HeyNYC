import subprocess
from pathlib import Path


def test_demo_launcher_uses_stable_domain_and_production_sender():
    script = Path(__file__).parents[1] / "scripts" / "serve_demo.sh"
    text = script.read_text()

    assert "nonepiscopal" not in text  # deployment specifics live in .env only (public-readiness B2)
    assert "+18882120042" not in text  # the real sender never ships in a tracked file (B1)
    assert "TWILIO_WHATSAPP_FROM" in text  # required by name in the env gate
    assert 'export HEYNYC_MODEL=' not in text  # RULED 2026-07-21: .env is the ONLY model source;
    # the old script export was silently OVERWRITTEN by .env sourcing anyway (set -a re-assigns),
    # so the luna pin never took effect and the pilot ran mini. The launcher now only VERIFIES.
    assert 'HEYNYC_MODEL' in text  # named in the required-env gate
    assert "--port 8791" in text
    assert " 8791" in text
    assert "127.0.0.1:8791/health" in text
    assert "set -m" not in text  # group kills under sh strike the script's own group (af7c044)
    assert "pkill -TERM -P" in text  # direct TERM + child sweep is the shutdown mechanism
    assert 'kill -TERM "$pid"' in text  # direct child kill, uv forwards to python (ec54aed)
    assert 'wait "$pid"' in text
    assert 'stop_one "${NGROK_PID:-}"' in text
    assert 'stop_one "${SERVER_PID:-}"' in text
    assert 'kill -0 "$NGROK_PID"' in text
    subprocess.run(["sh", "-n", script], check=True)


def test_demo_launcher_verifies_public_endpoint_and_required_env():
    """F059: the owner's pilot served a healthy local process while the tunnel had silently
    failed to bind the reserved domain. The launcher must verify the PUBLIC endpoint once after
    starting ngrok and fail loudly at the top with the names of any missing required env.
    F081: the launcher must NOT poll the public endpoint while supervising; its own 10-second
    public polling burned the ngrok free tier's monthly request quota (ERR_NGROK_727) and took
    the pilot down. Supervision watches local health + process liveness; the cron dead-man
    (health_watch.sh) owns the public endpoint at a low cadence."""
    script = Path(__file__).parents[1] / "scripts" / "serve_demo.sh"
    text = script.read_text()

    # Required-env gate, by name, before anything starts.
    for name in ("HEYNYC_PII_KEY", "HEYNYC_PII_SALT", "TWILIO_AUTH_TOKEN", "OPENAI_API_KEY"):
        assert name in text
    assert "missing" in text.lower()

    # One-shot public-endpoint gate after ngrok starts (this is the check that caught F081 live).
    assert "https://$DOMAIN/health" in text
    assert "public endpoint" in text.lower()

    # F081: the gate's failure path names the REAL ngrok error instead of guessing, from the
    # response header and the agent log (the TUI used to eat the error and garble the screen).
    assert "ngrok-error-code" in text
    assert "--log" in text and "ngrok.log" in text
    assert ">/dev/null" in text  # no TTY on stdout means no fullscreen TUI

    # F081: after the supervise loop starts, the public URL is never polled again.
    supervise = text[text.index("while kill -0") :]
    assert "https://$DOMAIN" not in supervise
    assert "127.0.0.1:8791/health" in supervise
    subprocess.run(["sh", "-n", script], check=True)


def test_health_watch_logs_public_endpoint_and_notifies_on_transitions():
    """F059 follow-up: the dead-man watcher checks the PUBLIC endpoint, appends a timestamped
    log line, and notifies only on down/up transitions, never every tick."""
    script = Path(__file__).parents[1] / "scripts" / "health_watch.sh"
    text = script.read_text()

    assert "https://$DOMAIN/health" in text
    assert "health.log" in text
    assert "health.state" in text
    assert "display notification" in text
    assert 'if [ "$status" = "FAIL" ] && [ "$prev" = "OK" ]' in text
    assert 'if [ "$status" = "OK" ] && [ "$prev" = "FAIL" ]' in text
    subprocess.run(["sh", "-n", script], check=True)


def test_demo_launcher_sources_env_itself():
    """`sh scripts/serve_demo.sh` must just work: the script loads the ignored .env itself
    when present, then still gates loudly on anything missing."""
    script = Path(__file__).parents[1] / "scripts" / "serve_demo.sh"
    text = script.read_text()

    assert ". ./.env" in text
    assert text.index(". ./.env") < text.index("Missing required env")
