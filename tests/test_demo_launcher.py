import subprocess
from pathlib import Path


def test_demo_launcher_uses_stable_domain_and_production_sender():
    script = Path(__file__).parents[1] / "scripts" / "serve_demo.sh"
    text = script.read_text()

    assert "nonepiscopal-inspiredly-sarai.ngrok-free.dev" in text
    assert "whatsapp:+18882120042" in text
    assert 'HEYNYC_MODEL="openai/gpt-5.4-mini"' in text
    assert "--port 8791" in text
    assert " 8791" in text
    assert "127.0.0.1:8791/health" in text
    assert "set -m" in text
    assert 'kill -TERM -- -"$pid"' in text
    assert 'wait "$pid"' in text
    assert 'stop_group "${NGROK_PID:-}"' in text
    assert 'stop_group "${SERVER_PID:-}"' in text
    assert 'kill -0 "$NGROK_PID"' in text
    subprocess.run(["sh", "-n", script], check=True)


def test_demo_launcher_verifies_public_endpoint_and_required_env():
    """F059: the owner's pilot served a healthy local process while the tunnel had silently
    failed to bind the reserved domain. The launcher must verify the PUBLIC endpoint after
    starting ngrok, keep verifying it while supervising, and fail loudly at the top with the
    names of any missing required env instead of trusting the parent shell."""
    script = Path(__file__).parents[1] / "scripts" / "serve_demo.sh"
    text = script.read_text()

    # Required-env gate, by name, before anything starts.
    for name in ("HEYNYC_PII_KEY", "HEYNYC_PII_SALT", "TWILIO_AUTH_TOKEN", "OPENAI_API_KEY"):
        assert name in text
    assert "missing" in text.lower()

    # Public-endpoint gate after ngrok starts, and continuous public verification while
    # supervising, so a dead tunnel kills the stack noisily instead of serving silence.
    assert text.count("https://$DOMAIN/health") >= 2
    assert "public endpoint" in text.lower()
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
