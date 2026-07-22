#!/bin/sh
# Dead-man watch for the supervised pilot (F059 follow-up): logs the PUBLIC endpoint's health
# and raises a desktop notification on the transition to down and again on recovery, so a
# silent outage can never again last hours unnoticed. This watcher is the ONLY public poller
# by design (F081: the launcher's 10s public polling exhausted the ngrok free tier's monthly
# request quota). Cron can fire it as often as it likes; the ACTUAL check cadence comes from
# `.env` via HEYNYC_HEALTH_WATCH_INTERVAL_S (default hourly), enforced by a last-check stamp.
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Cron starts with an empty environment; .env is the single config source.
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    . "$REPO_DIR/.env"
    set +a
fi

DOMAIN="${HEYNYC_NGROK_DOMAIN:-}"
INTERVAL="${HEYNYC_HEALTH_WATCH_INTERVAL_S:-3600}"
LOG="$REPO_DIR/.data/health.log"
STATE="$REPO_DIR/.data/health.state"
LAST="$REPO_DIR/.data/health.last"

mkdir -p "$REPO_DIR/.data"
now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
epoch="$(date +%s)"

[ -n "$DOMAIN" ] || { echo "$now SKIP no HEYNYC_NGROK_DOMAIN in .env" >> "$LOG"; exit 1; }

# Self-throttle: skip silently when the last real check is younger than the interval.
last="$(cat "$LAST" 2>/dev/null || echo 0)"
[ $((epoch - last)) -ge "$INTERVAL" ] || exit 0
echo "$epoch" > "$LAST"

if curl -fsS -m 10 "https://$DOMAIN/health" >/dev/null 2>&1; then
    status="OK"
else
    status="FAIL"
fi
echo "$now $status https://$DOMAIN/health" >> "$LOG"

prev="$(cat "$STATE" 2>/dev/null || echo "OK")"
echo "$status" > "$STATE"

notify() {
    /usr/bin/osascript -e "display notification \"$1\" with title \"HeyNYC pilot\"" 2>/dev/null || true
}

if [ "$status" = "FAIL" ] && [ "$prev" = "OK" ]; then
    notify "Public endpoint is DOWN. SMS and WhatsApp are not reachable."
fi
if [ "$status" = "OK" ] && [ "$prev" = "FAIL" ]; then
    notify "Public endpoint recovered."
fi
