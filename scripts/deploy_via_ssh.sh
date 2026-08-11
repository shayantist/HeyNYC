#!/bin/sh
# Run locally to start deployment on the configured private host over SSH
set -eu

[ "$#" -le 1 ] || { echo "usage: $0 [40-character pushed SHA]" >&2; exit 64; }
if [ "$#" -eq 1 ]; then
    case "$1" in
        *[!0-9a-f]*|'') echo "SHA must be 40 lowercase hexadecimal characters" >&2; exit 64 ;;
    esac
    [ "${#1}" -eq 40 ] || {
        echo "SHA must be 40 lowercase hexadecimal characters" >&2
        exit 64
    }
fi
ssh_host="${HEYNYC_DEPLOY_SSH_HOST:-heynyc-wsl}"
case "$ssh_host" in
    -*|''|*[[:space:]]*) echo "HEYNYC_DEPLOY_SSH_HOST must be one SSH host or alias" >&2; exit 64 ;;
esac

ssh "$ssh_host" wsl.exe -d Ubuntu --cd "~" --exec \
    git -C projects/HeyNYC pull --ff-only origin main
exec ssh -tt "$ssh_host" wsl.exe -d Ubuntu --cd "~" --exec \
    ./projects/HeyNYC/scripts/deploy.sh "$@"
