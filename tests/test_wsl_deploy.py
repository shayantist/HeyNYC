from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_wsl.sh"
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
        "esac\n",
    )
    _executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        f"echo \"git $*\" >> {log}\n"
        "if [ \"${1:-}\" = -C ]; then worktree=$2; shift 2; fi\n"
        "case \"$1 $2\" in\n"
        "  'worktree remove') target=$4; rm -rf \"$target\" ;;\n"
        "  'worktree add') target=$4; mkdir -p \"$target/.venv/bin\"; cat > \"$target/.venv/bin/python\" <<'PY'\n"
        "#!/bin/sh\n"
        "case \"$*\" in *'state_snapshot.py restore'*) while [ \"$#\" -gt 0 ]; do if [ \"$1\" = --target ]; then mkdir -p \"$2\"; break; fi; shift; done ;; esac\n"
        "exit 0\n"
        "PY\n"
        "chmod +x \"$target/.venv/bin/python\" ;;\n"
        "  'rev-parse --is-inside-work-tree') echo true ;;\n"
        "  'rev-parse HEAD') basename \"$worktree\" ;;\n"
        "  'status --porcelain') echo '?? .heynyc-ready' ;;\n"
        "  'show-ref --verify') [ \"$4\" = \"$EXPECTED_DEPLOY_REF\" ] ;;\n"
        "  'merge-base --is-ancestor') [ \"$4\" = \"$EXPECTED_DEPLOY_REF\" ] ;;\n"
        "esac\n",
    )
    _executable(fake_bin / "uv", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "curl", "#!/bin/sh\n[ \"${CURL_FAIL:-0}\" != 1 ]\n")
    _executable(
        fake_bin / "mv",
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = -Tf ]; then /bin/rm -f \"$3\"; exec /bin/mv \"$2\" \"$3\"; fi\n"
        "exec /bin/mv \"$@\"\n",
    )

    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HEYNYC_DEPLOY_ROOT": str(root),
        "HEYNYC_SOURCE_REPO": str(source),
        "STOP_COUNT": str(tmp_path / "stop-count"),
        "EXPECTED_DEPLOY_REF": "refs/remotes/origin/main",
    }
    return env, root, previous, log


def _run_deploy(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", SCRIPT, NEW_SHA], env=env, capture_output=True, text=True, check=False
    )


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

    sync = text.index("uv sync --frozen --extra whatsapp --extra pydantic-ai")
    stop = text.index('systemctl stop "$SERVICE"')
    snapshot = text.index('state_snapshot.py" create')
    restore_probe = text.index('state_snapshot.py" restore')
    switch = text.index('mv -Tf "$next_pointer" "$CURRENT"')
    start = text.index('systemctl start "$SERVICE"', switch)
    local_health = text.index("http://127.0.0.1:$PORT/health")
    public_health = text.index('https://$HEYNYC_NGROK_DOMAIN/health')
    reconcile = text.index("reconcile_twilio.py")

    assert sync < stop < snapshot < restore_probe < switch < start < local_health < public_health < reconcile
    assert "restore-check" in text
    assert "prestart_recovery" in text
    assert text.index("prestart_recovery=1") < stop


def test_wsl_deploy_fails_closed_and_parses_as_posix_shell() -> None:
    text = SCRIPT.read_text()

    assert "sudo -n true" in text
    assert "Automatic state rollback is intentionally disabled" in text
    assert text.count('systemctl stop "$SERVICE"') >= 2
    assert "TWILIO_FROM" in text and "TWILIO_WHATSAPP_FROM" in text
    assert '[ -L "$candidate/.env" ]' in text
    assert '[ -x "$candidate/.venv/bin/python" ]' in text
    assert "trap 'recover_before_start 129' HUP" in text
    assert "trap 'recover_before_start 130' INT" in text
    assert "trap 'recover_before_start 143' TERM" in text
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
    assert f"worktree remove --force {cached}" in commands
    assert f"worktree add --detach {cached} {NEW_SHA}" in commands
    assert (cached / ".heynyc-ready").read_text() == ""


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
