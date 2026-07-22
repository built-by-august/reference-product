"""Smoke tests for the CLI entrypoints (no TTY required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_uv(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "chi", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_hello_command_returns_result():
    proc = _run_uv(["hello", "Hello from CI"])
    assert proc.returncode == 0, proc.stderr
    assert "Hello from CI" in proc.stdout
    assert "trace_id" in proc.stdout


def test_run_help_exits_cleanly():
    proc = subprocess.run(
        [sys.executable, "-m", "chi_cli.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "hello" in proc.stdout
    assert "run" in proc.stdout
