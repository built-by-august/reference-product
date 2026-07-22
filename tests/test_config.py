"""Tests for CHI-1.3: structured config (YAML + env + defaults)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from chi_runtime.config import (
    Config,
    RetryConfig,
    load_config,
)
from chi_runtime.model import StubModelProvider


def test_defaults_are_sane() -> None:
    cfg = Config()
    assert cfg.retry.max_attempts == 3
    assert cfg.retry.base_delay == 0.5
    assert cfg.streaming.enabled is True
    assert cfg.observability.emit_metrics is True
    assert "stub" == cfg.model.provider


def test_yaml_loads_and_overrides_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            retry:
              max_attempts: 5
              base_delay: 1.0
            streaming:
              enabled: false
            observability:
              max_stored_runs: 50
            """
        )
    )
    cfg = load_config(str(cfg_file))
    assert cfg.retry.max_attempts == 5
    assert cfg.retry.base_delay == 1.0
    assert cfg.streaming.enabled is False
    assert cfg.observability.max_stored_runs == 50
    # Untouched defaults remain.
    assert cfg.retry.max_delay == RetryConfig().max_delay


def test_env_overrides_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHI_RETRY__MAX_ATTEMPTS", "7")
    monkeypatch.setenv("CHI_MODEL__MODEL", "gpt-4o-mini")
    monkeypatch.setenv("CHI_STREAM__ENABLED", "false")
    monkeypatch.setenv("CHI_OBS__EMIT_METRICS", "false")
    # A yaml providing different (lower-precedence) values.
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("retry:\n  max_attempts: 2\nmodel:\n  model: stub\n")
    cfg = load_config(str(cfg_file))
    assert cfg.retry.max_attempts == 7  # env wins over yaml
    assert cfg.model.model == "gpt-4o-mini"
    assert cfg.streaming.enabled is False
    assert cfg.observability.emit_metrics is False


def test_env_only_without_yaml(monkeypatch) -> None:
    monkeypatch.delenv("CHI_CONFIG_PATH", raising=False)
    monkeypatch.setenv("CHI_RETRY__MAX_ATTEMPTS", "4")
    cfg = load_config(None)
    assert cfg.retry.max_attempts == 4


def test_config_path_recorded(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("retry:\n  max_attempts: 9\n")
    cfg = load_config(str(cfg_file))
    assert cfg.config_path == str(cfg_file)


def test_stub_run_stream_yields_tokens() -> None:
    from chi_runtime.model import ModelTurn
    from chi_runtime.models import Message, Role

    turn = ModelTurn(messages=[Message(role=Role.USER, content="hi")], tools=[], model="stub")
    chunks = list(StubModelProvider().run_stream(turn))
    token_chunks = [c for c in chunks if c.event.value == "token"]
    done = [c for c in chunks if c.event.value == "done"]
    assert token_chunks, "expected at least one token chunk"
    assert done, "expected a final DONE chunk"
    assert done[0].usage is not None
    assert done[0].finish_reason.value == "stop"
