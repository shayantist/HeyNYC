#!/bin/sh
set -eu
set -m

cd "$(dirname "$0")/.."

DOMAIN="${HEYNYC_NGROK_DOMAIN:-nonepiscopal-inspiredly-sarai.ngrok-free.dev}"
export TWILIO_WHATSAPP_FROM="whatsapp:+18882120042"
export TWILIO_FROM="$TWILIO_WHATSAPP_FROM"
# RULED 2026-07-18: luna-medium is the benched production configuration (model-comparison.md);
# at low/none luna loses exactly the clock-reading and thread-holding that justify its cost
export HEYNYC_MODEL="openai/gpt-5.6-luna"
export HEYNYC_REASONING_EFFORT="medium"
export HEYNYC_SCOPE_MODEL="openai/gpt-5.4-mini"

# Load the ignored .env when present so `sh scripts/serve_demo.sh` just works; the parent
# shell's own exports still win because .env values only fill what sourcing sets.
if [ -f ./.env ]; then
    set -a
    . ./.env
    set +a
fi

# F059: fail loudly, by name, instead of trusting the shell to have every secret.
missing=""
[ -n "${HEYNYC_PII_KEY:-}" ] || missing="$missing HEYNYC_PII_KEY"
[ -n "${HEYNYC_PII_SALT:-}" ] || missing="$missing HEYNYC_PII_SALT"
[ -n "${TWILIO_AUTH_TOKEN:-}" ] || missing="$missing TWILIO_AUTH_TOKEN"
[ -n "${OPENAI_API_KEY:-}" ] || missing="$missing OPENAI_API_KEY"
if [ -n "$missing" ]; then
    echo "Missing required env:$missing" >&2
    echo "Load the ignored .env first, e.g.: set -a && . ./.env && set +a" >&2
    exit 1
fi

stop_group() {
    pid="$1"
    [ -z "$pid" ] || kill -TERM -- -"$pid" 2>/dev/null || true
    [ -z "$pid" ] || wait "$pid" 2>/dev/null || true
}

cleanup() {
    trap - EXIT INT TERM
    stop_group "${NGROK_PID:-}"
    stop_group "${SERVER_PID:-}"
}
trap cleanup EXIT INT TERM

uv run python -m heynyc serve --provider twilio --port 8791 &
SERVER_PID=$!

attempt=0
until curl -fsS http://127.0.0.1:8791/health >/dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || wait "$SERVER_PID"
    attempt=$((attempt + 1))
    [ "$attempt" -lt 30 ] || { echo "HeyNYC did not become healthy" >&2; exit 1; }
    sleep 1
done

ngrok http --url="https://$DOMAIN" 8791 &
NGROK_PID=$!

# F059: a running ngrok process is not a bound tunnel. Gate on the PUBLIC endpoint, the only
# check that proves Twilio can reach us (a stale-session claim collision serves 404 forever).
attempt=0
until curl -fsS "https://$DOMAIN/health" >/dev/null 2>&1; do
    kill -0 "$NGROK_PID" 2>/dev/null || { echo "ngrok exited before binding" >&2; exit 1; }
    attempt=$((attempt + 1))
    [ "$attempt" -lt 15 ] || {
        echo "public endpoint https://$DOMAIN/health never came up; is the reserved domain still claimed by an old session?" >&2
        exit 1
    }
    sleep 2
done

echo "HeyNYC: https://$DOMAIN"
echo "Twilio webhook: https://$DOMAIN/webhook/twilio"
public_failures=0
while kill -0 "$SERVER_PID" 2>/dev/null && kill -0 "$NGROK_PID" 2>/dev/null; do
    sleep 10
    if curl -fsS -m 8 "https://$DOMAIN/health" >/dev/null 2>&1; then
        public_failures=0
    else
        public_failures=$((public_failures + 1))
        if [ "$public_failures" -ge 3 ]; then
            echo "public endpoint dead for three straight checks; shutting down loudly" >&2
            exit 1
        fi
    fi
done
if ! kill -0 "$NGROK_PID" 2>/dev/null; then
    echo "ngrok stopped; shutting down HeyNYC" >&2
    exit 1
fi
wait "$SERVER_PID"
