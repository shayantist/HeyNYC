#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

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
[ -n "${HEYNYC_MODEL:-}" ] || missing="$missing HEYNYC_MODEL"
[ -n "${TWILIO_WHATSAPP_FROM:-}" ] || missing="$missing TWILIO_WHATSAPP_FROM"
[ -n "${HEYNYC_NGROK_DOMAIN:-}" ] || missing="$missing HEYNYC_NGROK_DOMAIN"
export TWILIO_FROM="${TWILIO_FROM:-${TWILIO_WHATSAPP_FROM:-}}"
DOMAIN="$HEYNYC_NGROK_DOMAIN"
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

# A stale ngrok from an earlier run holds the reserved domain and the new agent dies with
# ERR_NGROK_334 (observed live after the pre-fix Ctrl-C orphan). This launcher owns the pilot
# tunnel on this machine: clear any leftover agent before claiming.
if pgrep -x ngrok >/dev/null 2>&1; then
    echo "stale ngrok agent found; stopping it to reclaim $DOMAIN" >&2
    pkill -x ngrok || true
    sleep 2
fi
# F081: log to a file and detach stdout so ngrok never draws its fullscreen TUI over the
# launcher's own output (the TUI garbled the screen and ate the real error during the outage).
NGROK_LOG=".data/ngrok.log"
mkdir -p .data
: > "$NGROK_LOG"
ngrok http --url="https://$DOMAIN" 8791 --log "$NGROK_LOG" --log-format=logfmt >/dev/null 2>&1 &
NGROK_PID=$!

# F059: a running ngrok process is not a bound tunnel. Gate ONCE on the PUBLIC endpoint, the only
# check that proves Twilio can reach us (a stale-session claim collision serves 404 forever).
attempt=0
until curl -fsS "https://$DOMAIN/health" >/dev/null 2>&1; do
    kill -0 "$NGROK_PID" 2>/dev/null || { echo "ngrok exited before binding, last log lines:" >&2; tail -5 "$NGROK_LOG" >&2; exit 1; }
    attempt=$((attempt + 1))
    [ "$attempt" -lt 15 ] || {
        # F081: name the REAL error instead of guessing. ngrok's edge answers every blocked
        # request with an ngrok-error-code header (the outage was ERR_NGROK_727, monthly
        # request quota exhausted, which no amount of session-clearing would have fixed).
        code=$(curl -s -o /dev/null -D - -m 8 "https://$DOMAIN/health" | tr -d '\r' | awk 'tolower($1)=="ngrok-error-code:"{print $2}')
        echo "public endpoint https://$DOMAIN/health never came up${code:+ (ngrok says: $code, see https://ngrok.com/docs/errors)}" >&2
        tail -5 "$NGROK_LOG" >&2
        exit 1
    }
    sleep 2
done

echo "HeyNYC: https://$DOMAIN"
echo "Twilio webhook: https://$DOMAIN/webhook/twilio"
# F081: supervision polls LOCAL health only. Polling the public URL every 10s (8,640
# requests/day) burned the ngrok free tier's monthly quota and took the pilot down with
# ERR_NGROK_727. The cron dead-man (scripts/health_watch.sh) owns the public endpoint at a
# low cadence; here a dead tunnel is caught by ngrok process liveness.
local_failures=0
while kill -0 "$SERVER_PID" 2>/dev/null && kill -0 "$NGROK_PID" 2>/dev/null; do
    sleep 10
    if curl -fsS -m 8 http://127.0.0.1:8791/health >/dev/null 2>&1; then
        local_failures=0
    else
        local_failures=$((local_failures + 1))
        if [ "$local_failures" -ge 3 ]; then
            echo "local health dead for three straight checks; shutting down loudly" >&2
            exit 1
        fi
    fi
done
if ! kill -0 "$NGROK_PID" 2>/dev/null; then
    echo "ngrok stopped; shutting down HeyNYC" >&2
    exit 1
fi
wait "$SERVER_PID"
