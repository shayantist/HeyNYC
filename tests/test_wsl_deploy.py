from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy.sh"
LOCAL_DEPLOY_SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_via_ssh.sh"
NEW_SHA = "2" * 40
OLD_SHA = "1" * 40


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _deploy_fixture(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    root = tmp_path / "service"
    shared = root / "shared"
    releases = root / "releases"
    previous = releases / OLD_SHA
    source = tmp_path / "source"
    fake_bin = tmp_path / "bin"
    log = tmp_path / "commands.log"
    for directory in (shared / "data", previous / ".venv" / "bin", source, fake_bin):
        directory.mkdir(parents=True)
    old_index = shared / "data" / "index.lance"
    old_index.mkdir()
    (old_index / "old.txt").write_text("old index")
    env_file = shared / ".env"
    env_file.write_text(
        f"HEYNYC_DATA_DIR={shared / 'data'}\n"
        "HEYNYC_NGROK_DOMAIN=pilot.example\n"
        "TWILIO_ACCOUNT_SID=ACtest\n"
        "TWILIO_AUTH_TOKEN=test-token\n"
        "TWILIO_FROM=+18882120042\n"
    )
    (previous / ".heynyc-ready").touch()
    (previous / ".env").symlink_to(env_file)
    _executable(previous / ".venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")
    (root / "current").symlink_to(previous)

    _executable(fake_bin / "flock", "#!/bin/sh\nexit 0\n")
    _executable(
        fake_bin / "sudo",
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = -n ]; then shift; fi\n"
        "exec \"$@\"\n",
    )
    _executable(
        fake_bin / "systemctl",
        "#!/bin/sh\n"
        f"echo \"systemctl $*\" >> {log}\n"
        "case \"$1 $2\" in\n"
        f"  'show -p') case \"$*\" in *WorkingDirectory*) echo '{root / 'current'}' ;;"
        f" *ExecStart*) echo \"{{ path={root / 'current'}/.venv/bin/python ; argv[]={root / 'current'}/.venv/bin/python -m heynyc serve --provider twilio --port 8791${{SYSTEMD_EXTRA:-}} ; ignore_errors=no ; }}\" ;; esac ;;\n"
        "  'stop heynyc') count=0; [ ! -f \"$STOP_COUNT\" ] || count=$(cat \"$STOP_COUNT\"); count=$((count + 1)); echo \"$count\" > \"$STOP_COUNT\"; [ \"${FAIL_STOP_CALL:-}\" != \"$count\" ] ;;\n"
        "  'start heynyc') if [ \"${SIGNAL_ON_START:-0}\" = 1 ] && [ ! -f \"$SIGNAL_SENT\" ]; then : > \"$SIGNAL_SENT\"; kill -TERM \"$PPID\"; fi ;;\n"
        "esac\n",
    )
    _executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        f"echo \"git $*\" >> {log}\n"
        "if [ \"${1:-}\" = -C ]; then worktree=$2; shift 2; fi\n"
        "case \"$1 $2\" in\n"
        "  'worktree add') target=$4; mkdir -p \"$target/.venv/bin\" \"$target/scripts\"; if [ \"${TARGET_OLD_CONTROLLER:-0}\" = 1 ]; then printf '#!/bin/sh\\n# HEYNYC_PREPARED_RELEASE\\nexit 0\\n' > \"$target/scripts/deploy.sh\"; else cp \"$DEPLOY_SCRIPT\" \"$target/scripts/deploy.sh\"; fi; chmod +x \"$target/scripts/deploy.sh\"; cat > \"$target/.venv/bin/python\" <<'PY'\n"
        "#!/bin/sh\n"
        f"echo \"python $*\" >> {log}\n"
        "case \"$*\" in *'state_snapshot.py restore'*) while [ \"$#\" -gt 0 ]; do if [ \"$1\" = --target ]; then mkdir -p \"$2\"; break; fi; shift; done ;; esac\n"
        "case \"$*\" in *'-m heynyc index-build'*) if [ \"${SYMLINK_FRESH_INDEX:-0}\" = 1 ]; then mkdir -p \"$HEYNYC_DATA_DIR/real-index.lance\"; echo new > \"$HEYNYC_DATA_DIR/real-index.lance/new.txt\"; ln -s \"$HEYNYC_DATA_DIR/real-index.lance\" \"$HEYNYC_DATA_DIR/index.lance\"; else mkdir -p \"$HEYNYC_DATA_DIR/index.lance\"; echo new > \"$HEYNYC_DATA_DIR/index.lance/new.txt\"; fi; if [ \"${INCOMPLETE_INDEX_BUILD:-0}\" = 1 ]; then echo '  ok=1  chunks=1  failed=1'; else echo '  ok=2  chunks=2  failed=0'; fi ;; esac\n"
        "case \"$*\" in *'-m heynyc index-search'*) if [ \"${URL_ONLY_IN_TEXT:-0}\" = 1 ]; then echo 'document text mentions https://a858-nycnotify.nyc.gov/Home/FAQ and notify-nyc-short-code-terms-conditions-privacy-policy-information.page'; else echo 'https://a858-nycnotify.nyc.gov/Home/FAQ'; [ \"${FAIL_INDEX_PROBE:-0}\" = 1 ] || echo 'https://www.nyc.gov/site/em/resources/notify_nyc/notify-nyc-short-code-terms-conditions-privacy-policy-information.page'; fi ;; esac\n"
        "if [ \"${1:-}\" = -c ]; then echo 30; fi\n"
        "exit 0\n"
        "PY\n"
        "chmod +x \"$target/.venv/bin/python\" ;;\n"
        "  'rev-parse --is-inside-work-tree') echo true ;;\n"
        "  'rev-parse refs/remotes/origin/main') echo \"$DEFAULT_DEPLOY_SHA\" ;;\n"
        "  'rev-parse HEAD') basename \"$worktree\" | cut -c1-40 ;;\n"
        "  'status --porcelain') echo '?? .heynyc-ready' ;;\n"
        "  'show-ref --verify') [ \"$4\" = \"$EXPECTED_DEPLOY_REF\" ] ;;\n"
        "  'merge-base --is-ancestor') [ \"$4\" = \"$EXPECTED_DEPLOY_REF\" ] ;;\n"
        "esac\n",
    )
    _executable(fake_bin / "uv", "#!/bin/sh\nexit 0\n")
    _executable(
        fake_bin / "systemd-tmpfiles",
        "#!/bin/sh\n"
        f"echo \"systemd-tmpfiles $*\" >> {log}\n",
    )
    _executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "curl", "#!/bin/sh\n[ \"${CURL_FAIL:-0}\" != 1 ]\n")
    _executable(
        fake_bin / "mv",
        "#!/bin/sh\n"
        "case \"${1:-}\" in */shared/data/index.lance) [ \"${FAIL_OLD_INDEX_MOVE:-0}\" != 1 ] || exit 1 ;; esac\n"
        "case \"${1:-}\" in */new-data/index.lance) [ \"${FAIL_FRESH_INDEX_MOVE:-0}\" != 1 ] || exit 1 ;; esac\n"
        "if [ \"${1:-}\" = -Tf ]; then /usr/bin/python3 -c 'import os,sys; os.replace(sys.argv[1], sys.argv[2])' \"$2\" \"$3\"; if [ \"${SIGNAL_AFTER_SWITCH:-0}\" = 1 ] && [ ! -f \"$SIGNAL_SENT\" ]; then : > \"$SIGNAL_SENT\"; kill -TERM \"$PPID\"; fi; exit 0; fi\n"
        "exec /bin/mv \"$@\"\n",
    )
    _executable(
        fake_bin / "shasum",
        "#!/bin/sh\n"
        "case \"$*\" in *'-c ../SHA256SUMS'*) [ \"${FAIL_INDEX_CHECKSUM:-0}\" != 1 ] || exit 1 ;; esac\n"
        "exec /usr/bin/shasum \"$@\"\n",
    )
    _executable(
        fake_bin / "stat",
        "#!/bin/sh\n"
        "case \"$*\" in *shared/data*) echo 1 ;; *to-delete*) if [ \"${CROSS_FILESYSTEM_INDEX:-0}\" = 1 ]; then echo 2; else echo 1; fi ;; *service) echo 1 ;; *) exec /usr/bin/stat \"$@\" ;; esac\n",
    )

    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HEYNYC_DEPLOY_ROOT": str(root),
        "HEYNYC_SOURCE_REPO": str(source),
        "STOP_COUNT": str(tmp_path / "stop-count"),
        "SIGNAL_SENT": str(tmp_path / "signal-sent"),
        "EXPECTED_DEPLOY_REF": "refs/remotes/origin/main",
        "DEFAULT_DEPLOY_SHA": NEW_SHA,
        "HEYNYC_TMPFILES_CONFIG": str(tmp_path / "heynyc-backups.conf"),
        "DEPLOY_SCRIPT": str(SCRIPT),
    }
    return env, root, previous, log


def _run_deploy(
    env: dict[str, str], sha: str | None = NEW_SHA
) -> subprocess.CompletedProcess[str]:
    command = ["sh", SCRIPT]
    if sha is not None:
        command.append(sha)
    return subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )


def test_wsl_deploy_defaults_to_fetched_origin_main(tmp_path: Path) -> None:
    env, _, _, log = _deploy_fixture(tmp_path)

    result = _run_deploy(env, sha=None)

    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert "fetch --prune origin" in commands
    assert f"cat-file -e {NEW_SHA}^{{commit}}" in commands
    assert "worktree add --detach" in commands
    assert NEW_SHA in commands


def test_wsl_deploy_finds_uv_in_the_standard_user_install(tmp_path: Path) -> None:
    env, _, _, _ = _deploy_fixture(tmp_path)
    user_bin = tmp_path / ".local" / "bin"
    user_bin.mkdir(parents=True)
    Path(env["PATH"].split(":", 1)[0], "uv").rename(user_bin / "uv")

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr


def test_deploy_via_ssh_uses_only_a_local_ssh_alias(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "ssh.log"
    _executable(
        fake_bin / "ssh",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$SSH_LOG\"\n",
    )
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SSH_LOG": str(log),
    }

    result = subprocess.run(
        ["sh", LOCAL_DEPLOY_SCRIPT, NEW_SHA],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = log.read_text().splitlines()
    assert commands == [
        "heynyc-wsl wsl.exe -d Ubuntu --cd ~ --exec git -C projects/HeyNYC pull --ff-only origin main",
        f"-tt heynyc-wsl wsl.exe -d Ubuntu --cd ~ --exec ./projects/HeyNYC/scripts/deploy.sh {NEW_SHA}",
    ]
    text = LOCAL_DEPLOY_SCRIPT.read_text()
    assert "HostName " not in text
    assert "User " not in text
    assert "IdentityFile" not in text
    assert "StrictHostKeyChecking" not in text
    assert ".env" not in text
    assert "pwd" not in text
    assert "wsl_home" not in text
    assert LOCAL_DEPLOY_SCRIPT.stat().st_mode & 0o100


def test_deploy_via_ssh_deploys_latest_main_without_a_sha(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "ssh.log"
    _executable(fake_bin / "ssh", "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SSH_LOG\"\n")
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SSH_LOG": str(log),
    }

    result = subprocess.run(
        ["sh", LOCAL_DEPLOY_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines()[-1] == (
        "-tt heynyc-wsl wsl.exe -d Ubuntu --cd ~ --exec "
        "./projects/HeyNYC/scripts/deploy.sh"
    )


def test_deploy_via_ssh_rejects_an_ssh_option_as_the_host(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HEYNYC_DEPLOY_SSH_HOST": "-oProxyCommand=unexpected",
    }

    result = subprocess.run(
        ["sh", LOCAL_DEPLOY_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 64
    assert "HEYNYC_DEPLOY_SSH_HOST must be one SSH host or alias" in result.stderr


def test_deploy_via_ssh_rejects_command_text_as_the_sha(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "ssh.log"
    _executable(fake_bin / "ssh", f"#!/bin/sh\necho reached >> {log}\n")
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)}

    result = subprocess.run(
        ["sh", LOCAL_DEPLOY_SCRIPT, f"{NEW_SHA};whoami"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 64
    assert "SHA must be 40 lowercase hexadecimal characters" in result.stderr
    assert not log.exists()


def test_wsl_deploy_is_exact_sha_locked_and_uses_shared_state() -> None:
    text = SCRIPT.read_text()

    assert "flock" in text
    assert "origin/main" in text
    assert "merge-base --is-ancestor" in text
    assert "releases" in text and "current" in text and "shared" in text
    assert 'ln -s "$SHARED/.env" "$release/.env"' in text
    assert "HEYNYC_DATA_DIR" in text
    assert "cat .env" not in text
    assert 'systemctl show -p WorkingDirectory --value "$SERVICE"' in text
    assert 'systemctl show -p ExecStart --value "$SERVICE"' in text
    assert '[ -L "$CURRENT" ]' in text
    assert "--provider twilio --port $PORT" in text
    assert 'argv[]=$expected_exec ; ' in text
    assert '*"$expected_exec"*' not in text
    assert '[ -f "$candidate/.heynyc-ready" ]' in text
    assert '[ -x "$candidate/.venv/bin/python" ]' in text
    assert 'git -C "$candidate" rev-parse --is-inside-work-tree' in text


def test_wsl_deploy_prepares_before_the_short_stopped_window() -> None:
    text = SCRIPT.read_text()

    sync = text.index("sync --frozen --extra whatsapp --extra pydantic-ai")
    stop = text.index('systemctl stop "$SERVICE"')
    snapshot = text.index('state_snapshot.py" create')
    application_check = text.index('state_snapshot.py" verify')
    switch = text.index('mv -Tf "$next_pointer" "$CURRENT"')
    start = text.index('systemctl start "$SERVICE"', switch)
    local_health = text.index("http://127.0.0.1:$PORT/health")
    public_health = text.index('https://$HEYNYC_NGROK_DOMAIN/health')
    reconcile = text.index("reconcile_twilio.py")

    assert sync < stop < snapshot < application_check < switch < start < local_health < public_health < reconcile
    assert "--application-state" in text
    assert "restore-check" not in text
    assert "prestart_recovery" in text
    assert text.index("prestart_recovery=1") < stop


def test_wsl_deploy_builds_and_probes_a_fresh_index_before_stopping() -> None:
    text = SCRIPT.read_text()

    build = text.index("-m heynyc index-build")
    faq_probe = text.index("Notify NYC mobile app cost")
    terms_probe = text.index("Notify NYC short code message data rates")
    stop = text.index('systemctl stop "$SERVICE"')
    snapshot = text.index('state_snapshot.py" create')
    swap = text.index('mv "$fresh_index" "$active_index"')

    assert build < faq_probe < terms_probe < stop < snapshot < swap
    assert 'failed=0' in text
    assert '--urls-only' in text
    assert '"$ROOT/to-delete"' in text


def test_wsl_deploy_quarantines_the_old_index_and_activates_the_fresh_one(
    tmp_path: Path,
) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr
    assert (root / "shared" / "data" / "index.lance" / "new.txt").read_text() == "new\n"
    quarantines = list((root / "to-delete").glob("*-index-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "old-index.lance" / "old.txt").read_text() == "old index"
    assert (quarantines[0] / "inventory.tsv").is_file()
    assert (quarantines[0] / "SHA256SUMS").is_file()
    commands = log.read_text()
    assert commands.index("python -m heynyc index-build") < commands.index("systemctl stop heynyc")


def test_wsl_deploy_restores_the_old_index_on_prestart_failure(tmp_path: Path) -> None:
    env, root, _, _ = _deploy_fixture(tmp_path)
    env["SIGNAL_AFTER_SWITCH"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").read_text() == "old index"
    quarantines = list((root / "to-delete").glob("*-index-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "failed-index.lance" / "new.txt").read_text() == "new\n"


def test_wsl_deploy_hands_control_to_the_prepared_target_release() -> None:
    text = SCRIPT.read_text()

    assert 'HEYNYC_PREPARED_RELEASE' in text
    assert 'exec "$release/scripts/deploy.sh" "$sha"' in text
    assert '"$release/scripts/deploy.sh" --protocol' in text
    assert '"$ROOT/deploy.lock" -ef /proc/self/fd/9' in text
    assert 'flock -n 9' in text
    assert 'mktemp -d "$ROOT/to-delete/' in text
    assert 'mktemp -d "$ROOT/to-delete/$timestamp-$sha-rollback-' in text


def test_wsl_deploy_rejects_an_old_target_controller(tmp_path: Path) -> None:
    env, _, _, log = _deploy_fixture(tmp_path)
    env["TARGET_OLD_CONTROLLER"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert "target deploy controller does not support prepared releases" in result.stderr
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_an_incomplete_index_before_downtime(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["INCOMPLETE_INDEX_BUILD"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_a_failed_index_probe_before_downtime(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["FAIL_INDEX_PROBE"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_source_urls_found_only_in_document_text(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["URL_ONLY_IN_TEXT"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_cross_filesystem_index_moves_before_downtime(
    tmp_path: Path,
) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["CROSS_FILESYSTEM_INDEX"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert "index quarantine must share the live data filesystem" in result.stderr
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_a_symlinked_live_index_before_downtime(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    active = root / "shared" / "data" / "index.lance"
    target = root / "shared" / "data" / "real-index.lance"
    active.rename(target)
    active.symlink_to(target)

    result = _run_deploy(env)

    assert result.returncode != 0
    assert "active retrieval index must be a real directory" in result.stderr
    assert target.joinpath("old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_external_prepared_release_state(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["HEYNYC_PREPARED_RELEASE"] = str(root / "releases" / f"{NEW_SHA}-injected")

    result = _run_deploy(env)

    assert result.returncode == 64
    assert "prepared release requires inherited deployment state" in result.stderr
    assert not log.exists()


def test_wsl_deploy_ignores_prepared_release_state_from_dotenv(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    injected = root / "releases" / f"{NEW_SHA}-injected"
    with (root / "shared" / ".env").open("a") as handle:
        handle.write(f"HEYNYC_PREPARED_RELEASE={injected}\nHEYNYC_DEPLOY_LOCKED=1\n")

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr
    assert "worktree add --detach" in log.read_text()


def test_wsl_deploy_rejects_internal_variable_collisions_from_dotenv(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    with (root / "shared" / ".env").open("a") as handle:
        handle.write(f"sha={OLD_SHA}\n")

    result = _run_deploy(env)

    assert result.returncode != 0
    assert not log.exists()
    assert (root / "current").resolve().name == OLD_SHA


def test_wsl_deploy_rejects_a_symlinked_quarantine_root(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    quarantine_target = tmp_path / "external-quarantine"
    quarantine_target.mkdir()
    (root / "to-delete").symlink_to(quarantine_target)

    result = _run_deploy(env)

    assert result.returncode != 0
    assert "to-delete must be a real directory" in result.stderr
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_rejects_a_symlinked_fresh_index_before_downtime(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["SYMLINK_FRESH_INDEX"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert "fresh retrieval index must be a real directory" in result.stderr
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl stop heynyc" not in log.read_text()


def test_wsl_deploy_restores_the_old_index_after_checksum_failure(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["FAIL_INDEX_CHECKSUM"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl start heynyc" in log.read_text(), result.stderr


def test_wsl_deploy_restores_the_old_index_after_fresh_move_failure(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["FAIL_FRESH_INDEX_MOVE"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl start heynyc" in log.read_text(), result.stderr


def test_wsl_deploy_restarts_old_service_after_old_index_move_failure(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["FAIL_OLD_INDEX_MOVE"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "shared" / "data" / "index.lance" / "old.txt").is_file()
    assert "systemctl start heynyc" in log.read_text(), result.stderr


def test_wsl_deploy_installs_native_snapshot_retention(tmp_path: Path) -> None:
    env, _, _, log = _deploy_fixture(tmp_path)
    config = tmp_path / "heynyc-backups.conf"
    env["HEYNYC_TMPFILES_CONFIG"] = str(config)

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr
    assert config.read_text() == (
        f"d {tmp_path / 'service' / 'backups'} 0700 - - mM:30d -\n"
    )
    commands = log.read_text()
    assert "systemctl enable --now systemd-tmpfiles-clean.timer" in commands
    assert "systemd-tmpfiles --clean" in commands


def test_wsl_deploy_rejects_a_dangling_retention_config_symlink(tmp_path: Path) -> None:
    env, _, _, _ = _deploy_fixture(tmp_path)
    config = tmp_path / "heynyc-backups.conf"
    target = tmp_path / "unexpected-target"
    config.symlink_to(target)

    result = _run_deploy(env)

    assert result.returncode != 0
    assert not target.exists()


def test_wsl_deploy_fails_closed_and_parses_as_posix_shell() -> None:
    text = SCRIPT.read_text()

    assert "sudo -v" in text
    assert "sudo -n true" in text
    assert "Automatic state rollback is intentionally disabled" in text
    assert text.count('systemctl stop "$SERVICE"') >= 2
    assert "TWILIO_FROM" in text and "TWILIO_WHATSAPP_FROM" in text
    assert '[ -L "$candidate/.env" ]' in text
    assert '[ -x "$candidate/.venv/bin/python" ]' in text
    assert "trap 'recover_before_start 129' HUP" in text
    assert "trap 'recover_before_start 130' INT" in text
    assert "trap 'recover_before_start 143' TERM" in text
    assert "rm -rf" not in text
    assert "rm -f" not in text
    assert "worktree remove" not in text
    assert '--deletion-generation "$SHARED/data/.deletion-generation"' in text
    subprocess.run(["sh", "-n", SCRIPT], check=True)


def test_wsl_deploy_rebuilds_a_cached_release(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    cached = root / "releases" / NEW_SHA
    (cached / ".venv" / "bin").mkdir(parents=True)
    (cached / ".heynyc-ready").write_text("untrusted cache")
    (cached / ".env").symlink_to(root / "shared" / ".env")
    _executable(cached / ".venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr
    commands = log.read_text()
    assert f"worktree add --detach {cached}-" in commands
    assert (cached / ".heynyc-ready").read_text() == "untrusted cache"
    rebuilt = [path for path in (root / "releases").iterdir() if path.name.startswith(f"{NEW_SHA}-")]
    assert len(rebuilt) == 1
    assert (rebuilt[0] / ".heynyc-ready").read_text() == ""


def test_wsl_deploy_is_idempotent_after_a_cached_release_collision(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    cached = root / "releases" / NEW_SHA
    (cached / ".venv" / "bin").mkdir(parents=True)
    (cached / ".heynyc-ready").write_text("untrusted cache")
    (cached / ".env").symlink_to(root / "shared" / ".env")
    _executable(cached / ".venv" / "bin" / "python", "#!/bin/sh\nexit 0\n")

    first = _run_deploy(env)
    second = _run_deploy(env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "already deployed" in second.stdout
    assert log.read_text().count("worktree add --detach") == 1


def test_wsl_deploy_signal_after_pointer_switch_restores_previous(tmp_path: Path) -> None:
    env, root, previous, log = _deploy_fixture(tmp_path)
    env["SIGNAL_AFTER_SWITCH"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "current").resolve() == previous
    assert "pre-start deployment failure" in result.stderr
    assert "systemctl start heynyc" in log.read_text()


def test_wsl_deploy_signal_during_startup_restarts_current_release(tmp_path: Path) -> None:
    env, root, previous, log = _deploy_fixture(tmp_path)
    env["SIGNAL_ON_START"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "current").resolve() != previous
    assert "ensured the current service is started" in result.stderr
    assert log.read_text().count("systemctl start heynyc") == 2


def test_wsl_deploy_can_verify_an_explicit_candidate_ref(tmp_path: Path) -> None:
    env, _, _, log = _deploy_fixture(tmp_path)
    env["HEYNYC_DEPLOY_REF"] = "origin/codex/pydantic-ai-refactor"
    env["EXPECTED_DEPLOY_REF"] = "refs/remotes/origin/codex/pydantic-ai-refactor"

    result = _run_deploy(env)

    assert result.returncode == 0, result.stderr
    assert (
        f"merge-base --is-ancestor {NEW_SHA} "
        "refs/remotes/origin/codex/pydantic-ai-refactor"
        in log.read_text()
    )


def test_wsl_deploy_rejects_a_local_only_candidate_ref(tmp_path: Path) -> None:
    env, _, _, _ = _deploy_fixture(tmp_path)
    env["HEYNYC_DEPLOY_REF"] = "refs/heads/codex/pydantic-ai-refactor"

    result = _run_deploy(env)

    assert result.returncode == 64
    assert "must name a pushed origin ref" in result.stderr


def test_wsl_deploy_recovers_when_stop_partially_fails(tmp_path: Path) -> None:
    env, root, previous, log = _deploy_fixture(tmp_path)
    env["FAIL_STOP_CALL"] = "1"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "current").resolve() == previous
    assert "systemctl stop heynyc" in log.read_text()
    assert "systemctl start heynyc" in log.read_text()
    assert "pre-start deployment failure" in result.stderr


def test_wsl_deploy_rejects_extra_systemd_arguments(tmp_path: Path) -> None:
    env, _, _, log = _deploy_fixture(tmp_path)
    env["SYSTEMD_EXTRA"] = " --unexpected"

    result = _run_deploy(env)

    assert result.returncode == 78
    assert "must start HeyNYC" in result.stderr
    assert "fetch" not in log.read_text()


def test_wsl_deploy_reports_a_failed_health_cleanup(tmp_path: Path) -> None:
    env, root, _, log = _deploy_fixture(tmp_path)
    env["CURL_FAIL"] = "1"
    env["FAIL_STOP_CALL"] = "2"

    result = _run_deploy(env)

    assert result.returncode != 0
    assert (root / "current").resolve() == root / "releases" / NEW_SHA
    assert "local health failed" in result.stderr
    assert "could not stop the unhealthy new service" in result.stderr
    assert log.read_text().count("systemctl stop heynyc") == 2
