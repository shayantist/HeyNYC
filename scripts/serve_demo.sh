#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

DOMAIN="${HEYNYC_NGROK_DOMAIN:-}"
# Load the ignored .env when present so `sh scripts/serve_demo.sh` just works; the parent
# shell's own exports still win because .env values only fill what sourcing sets.
export TWILIO_FROM="${TWILIO_FROM:-$TWILIO_WHATSAPP_FROM}"
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
[ -n "${HEYNYC_MODEL:-}" ] || missing="$missing HEYNYC_MODEL"
[ -n "${TWILIO_WHATSAPP_FROM:-}" ] || missing="$missing TWILIO_WHATSAPP_FROM"
[ -n "${HEYNYC_NGROK_DOMAIN:-}" ] || missing="$missing HEYNYC_NGROK_DOMAIN"
if [ -n "$missing" ]; then
    echo "Missing required env:$missing" >&2
    echo "Load the ignored .env first, e.g.: set -a && . ./.env && set +a" >&2
    exit 1
fi

stop_one() {
    # Direct TERM to the child (uv forwards signals to python), then sweep grandchildren.
    # Group kills (kill -- -pgid) under non-interactive sh can hit the script's OWN group
    # and die mid-cleanup, orphaning the server on port 8791 (observed live: Ctrl-C left
    # the server running until a manual kill -9 by port).
    pid="$1"
    [ -z "$pid" ] && return 0
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    trap - EXIT INT TERM
    stop_one "${NGROK_PID:-}"
    stop_one "${SERVER_PID:-}"
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
