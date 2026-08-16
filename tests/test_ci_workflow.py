"""Lint the CI workflow: it must parse as YAML and keep its shape + pinned versions.

Guards security-audit finding F2 (this repo's first CI). If a pin is bumped or the dependency gate is
dropped, this test flags it so the change is deliberate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")  # PyYAML is a project dependency

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_ci_workflow_is_well_formed_yaml():
    doc = _load()
    assert isinstance(doc, dict)
    assert doc["name"] == "CI"
    assert "jobs" in doc and isinstance(doc["jobs"], dict) and doc["jobs"]


def test_ci_triggers_on_push_and_pull_request():
    doc = _load()
    # PyYAML parses the bare key `on` as the boolean True (YAML 1.1), so accept either form.
    triggers = doc.get("on", doc.get(True))
    assert triggers is not None
    assert "push" in triggers and "pull_request" in triggers


def test_ci_runs_the_offline_suite_and_dependency_gate():
    text = WORKFLOW.read_text()
    assert "python -m pytest" in text                      # offline suite
    assert "pip-audit" in text                              # dependency CVE scan
    assert "uv sync" in text and "--extra dev" in text      # installs the dev extra
    assert "--extra pydantic-ai" in text                    # exercises the candidate runtime
    assert text.count("--extra browser") == 2               # test and audit deployed fallback
    # Never install the heavy nli extra in CI (torch + a git dep the offline suite never needs).
    assert "--extra nli" not in text and "--all-extras" not in text


def test_ci_action_and_tool_versions_are_pinned():
    text = WORKFLOW.read_text()
    for pin in (
        "actions/checkout@v5.0.0",
        "astral-sh/setup-uv@v8.3.2",
        "pypa/gh-action-pip-audit@v1.1.0",
    ):
        assert pin in text, pin
    # No unpinned floating refs on third-party actions.
    assert "@main" not in text and "@master" not in text


def test_ci_does_not_duplicate_github_native_secret_scanning():
    assert "trufflehog" not in WORKFLOW.read_text().lower()
