#!/bin/sh
# Dead-man watch for the supervised pilot (F059 follow-up): logs the PUBLIC endpoint's health
# and raises a desktop notification on the transition to down and again on recovery, so a
# silent outage can never again last hours unnoticed. Designed for cron, every 15 minutes:
# each tick spends one request of the ngrok free tier's monthly quota, and this watcher is the
# ONLY public poller by design (F081: the launcher's 10s public polling exhausted the quota).
set -eu

DOMAIN="${HEYNYC_NGROK_DOMAIN:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/.data/health.log"
STATE="$REPO_DIR/.data/health.state"

mkdir -p "$REPO_DIR/.data"
now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

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
