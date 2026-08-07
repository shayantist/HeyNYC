#!/bin/sh
# Manual, exact-SHA deployment for the single-host WSL pilot.
set -eu

[ "$#" -eq 1 ] || { echo "usage: $0 <40-character pushed SHA>" >&2; exit 64; }
sha="$1"
case "$sha" in
    *[!0-9a-f]*|'') echo "deployment SHA must be lowercase hexadecimal" >&2; exit 64 ;;
esac
[ "${#sha}" -eq 40 ] || { echo "deployment SHA must contain 40 characters" >&2; exit 64; }

ROOT="${HEYNYC_DEPLOY_ROOT:-$HOME/services/heynyc}"
SOURCE="${HEYNYC_SOURCE_REPO:-$HOME/projects/HeyNYC}"
SHARED="$ROOT/shared"
RELEASES="$ROOT/releases"
CURRENT="$ROOT/current"
BACKUPS="$ROOT/backups"
SERVICE="${HEYNYC_SYSTEMD_SERVICE:-heynyc}"
PORT="${HEYNYC_PORT:-8791}"

mkdir -p "$ROOT" "$RELEASES" "$BACKUPS"
exec 9>"$ROOT/deploy.lock"
flock -n 9 || { echo "another deployment holds $ROOT/deploy.lock" >&2; exit 75; }
sudo -n true 2>/dev/null || {
    echo "sudo authorization is not cached; run sudo -v interactively, then retry" >&2
    exit 77
}
[ -f "$SHARED/.env" ] || { echo "missing shared .env at $SHARED/.env" >&2; exit 66; }
[ -d "$SHARED/data" ] || { echo "missing shared data at $SHARED/data" >&2; exit 66; }

set -a
. "$SHARED/.env"
set +a
deploy_ref="${HEYNYC_DEPLOY_REF:-origin/main}"
case "$deploy_ref" in
    origin/*) ;;
    *) echo "HEYNYC_DEPLOY_REF must name a pushed origin ref" >&2; exit 64 ;;
esac
remote_ref="refs/remotes/$deploy_ref"
[ "${HEYNYC_DATA_DIR:-}" = "$SHARED/data" ] || {
    echo "HEYNYC_DATA_DIR must point to $SHARED/data" >&2
    exit 78
}
[ -n "${HEYNYC_NGROK_DOMAIN:-}" ] || { echo "HEYNYC_NGROK_DOMAIN is required" >&2; exit 78; }
[ -n "${TWILIO_ACCOUNT_SID:-}" ] || { echo "TWILIO_ACCOUNT_SID is required" >&2; exit 78; }
[ -n "${TWILIO_AUTH_TOKEN:-}" ] || { echo "TWILIO_AUTH_TOKEN is required" >&2; exit 78; }
[ -n "${TWILIO_FROM:-}${TWILIO_WHATSAPP_FROM:-}" ] || {
    echo "at least one Twilio recipient address is required" >&2
    exit 78
}

validate_release() {
    candidate="$1"
    expected_sha="${2:-}"
    case "$candidate" in
        "$RELEASES"/*) ;;
        *) return 1 ;;
    esac
    [ ! -L "$candidate" ] &&
        [ -d "$candidate" ] &&
        [ -f "$candidate/.heynyc-ready" ] &&
        [ -L "$candidate/.env" ] &&
        [ "$(readlink -f "$candidate/.env")" = "$SHARED/.env" ] &&
        [ -x "$candidate/.venv/bin/python" ] &&
        [ "$(git -C "$candidate" rev-parse --is-inside-work-tree 2>/dev/null)" = "true" ] &&
        git -C "$candidate" diff --quiet -- &&
        git -C "$candidate" diff --cached --quiet -- &&
        [ "$(git -C "$candidate" status --porcelain --untracked-files=all)" = "?? .heynyc-ready" ] &&
        { [ -z "$expected_sha" ] || [ "$(git -C "$candidate" rev-parse HEAD)" = "$expected_sha" ]; }
}

[ -L "$CURRENT" ] || { echo "$CURRENT must be a release symlink" >&2; exit 78; }
previous="$(readlink -f "$CURRENT")"
validate_release "$previous" || { echo "current release target is not a valid release" >&2; exit 78; }
working_directory="$(sudo systemctl show -p WorkingDirectory --value "$SERVICE")"
[ "$working_directory" = "$CURRENT" ] || {
    echo "$SERVICE must use WorkingDirectory=$CURRENT before release deployments" >&2
    exit 78
}
exec_start="$(sudo systemctl show -p ExecStart --value "$SERVICE")"
expected_exec="$CURRENT/.venv/bin/python -m heynyc serve --provider twilio --port $PORT"
case "$exec_start" in
    "{ path=$CURRENT/.venv/bin/python ; argv[]=$expected_exec ; "*) ;;
    *) echo "$SERVICE must start HeyNYC from $CURRENT on port $PORT" >&2; exit 78 ;;
esac

git -C "$SOURCE" fetch --prune origin
git -C "$SOURCE" cat-file -e "$sha^{commit}"
git -C "$SOURCE" show-ref --verify --quiet "$remote_ref" || {
    echo "$deploy_ref is not a fetched remote ref" >&2
    exit 65
}
git -C "$SOURCE" merge-base --is-ancestor "$sha" "$remote_ref" || {
    echo "$sha is not contained in $deploy_ref" >&2
    exit 65
}

release="$RELEASES/$sha"
if [ "$release" = "$previous" ]; then
    echo "$sha is already deployed"
    exit 0
fi
if [ -L "$release" ]; then
    rm -f "$release"
elif [ -e "$release" ]; then
    git -C "$SOURCE" worktree remove --force "$release"
fi
git -C "$SOURCE" worktree add --detach "$release" "$sha"
ln -s "$SHARED/.env" "$release/.env"
(cd "$release" && uv sync --frozen --extra whatsapp --extra pydantic-ai)
: > "$release/.heynyc-ready"
validate_release "$release" "$sha" || { echo "release directory is not ready for requested SHA" >&2; exit 78; }
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="$BACKUPS/$timestamp-$sha-$$"
restore_probe="$ROOT/.restore-check.$$"
python="$release/.venv/bin/python"
[ ! -e "$restore_probe" ] || { echo "restore probe target already exists" >&2; exit 73; }
next_pointer="$ROOT/.current.$$"
prestart_recovery=0
pointer_switched=0

recover_before_start() {
    status="$1"
    [ "$status" -ne 0 ] || status=1
    trap - EXIT HUP INT TERM
    if [ "$prestart_recovery" -eq 1 ]; then
        rollback_ok=1
        rm -rf "$restore_probe" 2>/dev/null || true
        rm -f "$next_pointer" 2>/dev/null || true
        if [ "$pointer_switched" -eq 1 ]; then
            rollback_pointer="$ROOT/.rollback.$$"
            if ! ln -s "$previous" "$rollback_pointer" 2>/dev/null; then
                rollback_ok=0
            elif ! mv -Tf "$rollback_pointer" "$CURRENT" 2>/dev/null; then
                rm -f "$rollback_pointer" 2>/dev/null || true
                rollback_ok=0
            fi
        fi
        current_target="$(readlink -f "$CURRENT" 2>/dev/null || true)"
        if [ "$current_target" != "$previous" ] || ! validate_release "$previous"; then
            rollback_ok=0
        fi
        if [ "$rollback_ok" -eq 1 ] && ! sudo systemctl start "$SERVICE" 2>/dev/null; then
            rollback_ok=0
        fi
        if [ "$rollback_ok" -eq 1 ]; then
            echo "pre-start deployment failure; restored the old pointer and service" >&2
        else
            echo "pre-start recovery failed; operator intervention is required" >&2
        fi
    fi
    exit "$status"
}
trap 'recover_before_start $?' EXIT
trap 'recover_before_start 129' HUP
trap 'recover_before_start 130' INT
trap 'recover_before_start 143' TERM

# The old release serves while Git and dependencies prepare. The stopped window begins here.
prestart_recovery=1
sudo systemctl stop "$SERVICE"
if ! "$python" "$release/scripts/state_snapshot.py" create \
    --data-dir "$SHARED/data" --output "$snapshot" --app-sha "$sha" --quiesced; then
    echo "snapshot failed before the release pointer changed" >&2
    exit 1
fi
if ! "$python" "$release/scripts/state_snapshot.py" restore "$snapshot" \
    --target "$restore_probe"; then
    echo "restore verification failed before the release pointer changed" >&2
    exit 1
fi
rm -rf "$restore_probe"

ln -s "$release" "$next_pointer"
mv -Tf "$next_pointer" "$CURRENT"
pointer_switched=1
prestart_recovery=0
trap - EXIT HUP INT TERM
if ! sudo systemctl start "$SERVICE"; then
    sudo systemctl stop "$SERVICE" 2>/dev/null || true
    echo "new service failed to start" >&2
    echo "Automatic state rollback is intentionally disabled after startup; snapshot: $snapshot" >&2
    echo "Previous release: ${previous:-none}" >&2
    exit 1
fi

health_check() {
    url="$1"
    attempts=0
    until curl -fsS --max-time 5 "$url" >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        [ "$attempts" -lt 30 ] || return 1
        sleep 1
    done
}

if ! health_check "http://127.0.0.1:$PORT/health"; then
    if ! sudo systemctl stop "$SERVICE"; then
        echo "could not stop the unhealthy new service" >&2
    fi
    echo "local health failed" >&2
    echo "Automatic state rollback is intentionally disabled after startup; snapshot: $snapshot" >&2
    echo "Previous release: ${previous:-none}" >&2
    exit 1
fi
if ! health_check "https://$HEYNYC_NGROK_DOMAIN/health"; then
    if ! sudo systemctl stop "$SERVICE"; then
        echo "could not stop the unhealthy new service" >&2
    fi
    echo "public health failed" >&2
    echo "Automatic state rollback is intentionally disabled after startup; snapshot: $snapshot" >&2
    echo "Previous release: ${previous:-none}" >&2
    exit 1
fi

printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sha" "$snapshot" \
    >> "$ROOT/deployments.tsv"

set -- --database "$SHARED/data/channels.sqlite3" --hours 24
[ -z "${TWILIO_FROM:-}" ] || set -- "$@" --to "$TWILIO_FROM"
[ -z "${TWILIO_WHATSAPP_FROM:-}" ] || set -- "$@" --to "$TWILIO_WHATSAPP_FROM"
set +e
"$python" "$release/scripts/reconcile_twilio.py" "$@"
reconcile_status=$?
set -e
if [ "$reconcile_status" -ne 0 ]; then
    echo "deployment is healthy, but Twilio SID reconciliation needs operator review" >&2
    exit "$reconcile_status"
fi

echo "deployed $sha"
echo "snapshot: $snapshot"
