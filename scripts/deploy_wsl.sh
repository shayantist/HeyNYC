#!/bin/sh
# Manual, exact-SHA deployment for the single-host WSL pilot.
set -eu

DEPLOY_PROTOCOL="heynyc-deploy-v2"
if [ "$#" -eq 1 ] && [ "$1" = --protocol ]; then
    echo "$DEPLOY_PROTOCOL"
    exit 0
fi
[ "$#" -eq 1 ] || { echo "usage: $0 <40-character pushed SHA>" >&2; exit 64; }
deploy_locked="${HEYNYC_DEPLOY_LOCKED:-0}"
prepared_release="${HEYNYC_PREPARED_RELEASE:-}"
unset HEYNYC_DEPLOY_LOCKED HEYNYC_PREPARED_RELEASE
[ -z "$prepared_release" ] || [ "$deploy_locked" = 1 ] || {
    echo "prepared release requires inherited deployment state" >&2
    exit 64
}
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
TMPFILES_CONFIG="${HEYNYC_TMPFILES_CONFIG:-/etc/tmpfiles.d/heynyc-backups.conf}"
SERVICE="${HEYNYC_SYSTEMD_SERVICE:-heynyc}"
PORT="${HEYNYC_PORT:-8791}"
readonly DEPLOY_PROTOCOL deploy_locked prepared_release sha ROOT SOURCE SHARED RELEASES CURRENT
readonly BACKUPS TMPFILES_CONFIG SERVICE PORT

mkdir -p "$ROOT" "$RELEASES" "$BACKUPS"
if [ "$deploy_locked" = 1 ]; then
    if { [ -e /proc/self/fd/9 ] && ! [ "$ROOT/deploy.lock" -ef /proc/self/fd/9 ]; } || ! flock -n 9; then
        echo "inherited deployment lock is unavailable" >&2
        exit 75
    fi
else
    exec 9>"$ROOT/deploy.lock"
    flock -n 9 || { echo "another deployment holds $ROOT/deploy.lock" >&2; exit 75; }
fi
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
    candidate_real="$(readlink -f "$candidate" 2>/dev/null)" || return 1
    releases_real="$(readlink -f "$RELEASES" 2>/dev/null)" || return 1
    case "$candidate" in
        "$RELEASES"/*) ;;
        *) return 1 ;;
    esac
    [ "$(dirname "$candidate_real")" = "$releases_real" ] &&
        [ "$candidate_real" = "$candidate" ] &&
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

filesystem_id() {
    stat -c %d "$1" 2>/dev/null || stat -f %d "$1"
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
if [ "$(git -C "$previous" rev-parse HEAD)" = "$sha" ]; then
    echo "$sha is already deployed"
    exit 0
fi
if [ -n "$prepared_release" ]; then
    release="$prepared_release"
    validate_release "$release" "$sha" || {
        echo "prepared release is not ready for requested SHA" >&2
        exit 78
    }
else
    if [ -e "$release" ] || [ -L "$release" ]; then
        release="$RELEASES/$sha-$(date -u +%Y%m%dT%H%M%SZ)-$$"
        [ ! -e "$release" ] && [ ! -L "$release" ] || {
            echo "unique release path already exists: $release" >&2
            exit 73
        }
    fi
    git -C "$SOURCE" worktree add --detach "$release" "$sha"
    ln -s "$SHARED/.env" "$release/.env"
    (cd "$release" && uv sync --frozen --extra whatsapp --extra pydantic-ai)
    : > "$release/.heynyc-ready"
    validate_release "$release" "$sha" || { echo "release directory is not ready for requested SHA" >&2; exit 78; }
    if [ "$("$release/scripts/deploy_wsl.sh" --protocol 2>/dev/null)" != "$DEPLOY_PROTOCOL" ]; then
        echo "target deploy controller does not support prepared releases" >&2
        exit 78
    fi
    export HEYNYC_PREPARED_RELEASE="$release"
    export HEYNYC_DEPLOY_LOCKED=1
    exec "$release/scripts/deploy_wsl.sh" "$sha"
fi
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="$BACKUPS/$timestamp-$sha-$$"
python="$release/.venv/bin/python"
next_pointer="$ROOT/.current.$timestamp.$$"
[ ! -e "$next_pointer" ] && [ ! -L "$next_pointer" ] || {
    echo "next release pointer already exists: $next_pointer" >&2
    exit 73
}
case "$BACKUPS" in
    *[[:space:]]*) echo "backup path must not contain whitespace" >&2; exit 78 ;;
esac
[ ! -L "$ROOT/to-delete" ] || { echo "to-delete must be a real directory" >&2; exit 78; }
mkdir -p "$ROOT/to-delete"
[ -d "$ROOT/to-delete" ] && [ ! -L "$ROOT/to-delete" ] || {
    echo "to-delete must be a real directory" >&2
    exit 78
}
index_quarantine="$(mktemp -d "$ROOT/to-delete/$timestamp-$sha-index-XXXXXX")"
index_build_data="$index_quarantine/new-data"
active_index="$SHARED/data/index.lance"
fresh_index="$index_build_data/index.lance"
[ ! -L "$active_index" ] && [ -d "$active_index" ] || {
    echo "active retrieval index must be a real directory" >&2
    exit 78
}
[ "$(filesystem_id "$active_index")" = "$(filesystem_id "$index_quarantine")" ] &&
    [ "$(filesystem_id "$ROOT")" = "$(filesystem_id "$index_quarantine")" ] || {
    echo "index quarantine must share the live data filesystem" >&2
    exit 78
}
command -v shasum >/dev/null 2>&1 || { echo "shasum is required" >&2; exit 69; }
mkdir -p "$index_build_data"
build_log="$index_quarantine/index-build.log"
if ! HEYNYC_DATA_DIR="$index_build_data" "$python" -m heynyc index-build >"$build_log" 2>&1; then
    cat "$build_log"
    echo "index build failed; evidence preserved at $index_quarantine" >&2
    exit 1
fi
cat "$build_log"
if ! grep -Eq 'ok=[1-9][0-9]*[[:space:]]+chunks=[1-9][0-9]*[[:space:]]+failed=0' "$build_log"; then
    echo "index build did not produce a complete nonempty corpus; evidence preserved at $index_quarantine" >&2
    exit 1
fi
if [ -L "$fresh_index" ] || [ ! -d "$fresh_index" ]; then
    echo "fresh retrieval index must be a real directory; evidence preserved at $index_quarantine" >&2
    exit 1
fi
faq_probe="$index_quarantine/notify-faq-probe.log"
HEYNYC_DATA_DIR="$index_build_data" "$python" -m heynyc index-search --urls-only \
    "Notify NYC mobile app cost money in-app purchases free" >"$faq_probe"
grep -Fxq 'https://a858-nycnotify.nyc.gov/Home/FAQ' "$faq_probe" || {
    echo "fresh index did not retrieve the Notify NYC FAQ; evidence preserved at $index_quarantine" >&2
    exit 1
}
terms_probe="$index_quarantine/notify-terms-probe.log"
HEYNYC_DATA_DIR="$index_build_data" "$python" -m heynyc index-search --urls-only \
    "Notify NYC short code message data rates wireless provider charges" >"$terms_probe"
grep -Fxq 'https://www.nyc.gov/site/em/resources/notify_nyc/notify-nyc-short-code-terms-conditions-privacy-policy-information.page' "$terms_probe" || {
    echo "fresh index did not retrieve the Notify NYC terms; evidence preserved at $index_quarantine" >&2
    exit 1
}
retention_days=$("$python" -c 'from heynyc.core.pii_crypto import retention_days; print(f"{retention_days():g}")')
tmpfiles_rule="d $BACKUPS 0700 - - mM:${retention_days}d -"
if [ -e "$TMPFILES_CONFIG" ] || [ -L "$TMPFILES_CONFIG" ]; then
    [ ! -L "$TMPFILES_CONFIG" ] && [ "$(cat "$TMPFILES_CONFIG")" = "$tmpfiles_rule" ] || {
        echo "unexpected snapshot retention config at $TMPFILES_CONFIG" >&2
        exit 78
    }
else
    printf '%s\n' "$tmpfiles_rule" | sudo tee "$TMPFILES_CONFIG" >/dev/null
fi
sudo systemctl enable --now systemd-tmpfiles-clean.timer
sudo systemd-tmpfiles --clean "$TMPFILES_CONFIG"
prestart_recovery=0
startup_attempted=0
index_change_started=0

restore_previous_index() {
    [ "$index_change_started" -eq 1 ] || return 0
    if [ -d "$index_quarantine/old-index.lance" ]; then
        if [ -e "$active_index" ] || [ -L "$active_index" ]; then
            mv "$active_index" "$index_quarantine/failed-index.lance" || return 1
        fi
        mv "$index_quarantine/old-index.lance" "$active_index"
    elif [ ! -d "$active_index" ]; then
        return 1
    fi
    return 0
}

recover_before_start() {
    status="$1"
    [ "$status" -ne 0 ] || status=1
    trap - EXIT HUP INT TERM
    if [ "$prestart_recovery" -eq 1 ]; then
        rollback_ok=1
        current_target="$(readlink -f "$CURRENT" 2>/dev/null || true)"
        if [ "$startup_attempted" -eq 0 ] && [ "$current_target" != "$previous" ]; then
            if ! rollback_directory="$(mktemp -d "$ROOT/to-delete/$timestamp-$sha-rollback-XXXXXX")"; then
                rollback_ok=0
            else
                rollback_pointer="$rollback_directory/current"
                if ! ln -s "$previous" "$rollback_pointer" 2>/dev/null; then
                    rollback_ok=0
                elif ! mv -Tf "$rollback_pointer" "$CURRENT" 2>/dev/null; then
                    echo "rollback pointer remains at $rollback_pointer" >&2
                    rollback_ok=0
                fi
            fi
        fi
        current_target="$(readlink -f "$CURRENT" 2>/dev/null || true)"
        if [ "$startup_attempted" -eq 0 ]; then
            if [ "$current_target" != "$previous" ] || ! validate_release "$previous"; then
                rollback_ok=0
            elif ! restore_previous_index; then
                rollback_ok=0
            fi
            if [ "$rollback_ok" -eq 1 ] && ! sudo systemctl start "$SERVICE" 2>/dev/null; then
                rollback_ok=0
            fi
        elif ! sudo systemctl start "$SERVICE" 2>/dev/null; then
            rollback_ok=0
        fi
        if [ "$rollback_ok" -eq 1 ]; then
            if [ "$startup_attempted" -eq 0 ]; then
                echo "pre-start deployment failure; restored the old pointer and service" >&2
            else
                echo "deployment interrupted during startup; ensured the current service is started" >&2
            fi
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
if ! "$python" "$release/scripts/state_snapshot.py" verify "$snapshot" \
    --application-state --deletion-generation "$SHARED/data/.deletion-generation"; then
    echo "snapshot application verification failed before the release pointer changed" >&2
    exit 1
fi
[ -d "$active_index" ] || {
    echo "active retrieval index is missing" >&2
    exit 1
}
index_change_started=1
mv "$active_index" "$index_quarantine/old-index.lance"
(
    cd "$index_quarantine/old-index.lance"
    : >../inventory.tsv
    : >../SHA256SUMS
    find . -type f | LC_ALL=C sort | while IFS= read -r file; do
        size=$(wc -c <"$file" | tr -d ' ')
        printf '%s\t%s\n' "${file#./}" "$size" >>../inventory.tsv
        shasum -a 256 "$file" >>../SHA256SUMS
    done
    shasum -a 256 -c ../SHA256SUMS >/dev/null
)
mv "$fresh_index" "$active_index"

ln -s "$release" "$next_pointer"
mv -Tf "$next_pointer" "$CURRENT"
startup_attempted=1
if ! sudo systemctl start "$SERVICE"; then
    prestart_recovery=0
    trap - EXIT HUP INT TERM
    sudo systemctl stop "$SERVICE" 2>/dev/null || true
    echo "new service failed to start" >&2
    echo "Automatic state rollback is intentionally disabled after startup; snapshot: $snapshot" >&2
    echo "Previous release: ${previous:-none}" >&2
    exit 1
fi
trap - EXIT HUP INT TERM
prestart_recovery=0

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
