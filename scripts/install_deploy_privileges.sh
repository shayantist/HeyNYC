#!/bin/sh
# One-time WSL bootstrap for unattended HeyNYC deployments
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

sudo -v
sudo visudo -cf "$SCRIPT_DIR/heynyc-deploy.sudoers"
sudo install -o root -g root -m 0755 \
    "$SCRIPT_DIR/heynyc-deploy-privileged" \
    /usr/local/sbin/heynyc-deploy-privileged
sudo install -o root -g root -m 0440 \
    "$SCRIPT_DIR/heynyc-deploy.sudoers" \
    /etc/sudoers.d/heynyc-deploy
sudo visudo -cf /etc/sudoers.d/heynyc-deploy
sudo -n /usr/local/sbin/heynyc-deploy-privileged check

echo "HeyNYC deployment privileges installed"
