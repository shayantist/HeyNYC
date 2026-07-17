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
