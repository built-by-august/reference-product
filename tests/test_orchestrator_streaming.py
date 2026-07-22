"""Tests for CHI-1.3: orchestrator streaming + metrics integration."""

from __future__ import annotations

from chi_runtime import Orchestrator, StubModelProvider
from chi_runtime.agents import build_hello_agent
from chi_runtime.config import get_config, reload_config
from chi_runtime.metrics import get_metrics_store
from chi_runtime.models import StreamEventType


def _count_runs() -> int:
    return get_metrics_store().snapshot().total_runs


def test_orchestrator_run_stream_yields_tokens_and_done() -> None:
    orch = Orchestrator(provider=StubModelProvider())
    chunks = list(orch.run_stream(build_hello_agent(), "hello there"))
    tokens = [c for c in chunks if c.event == StreamEventType.TOKEN]
    done = [c for c in chunks if c.event == StreamEventType.DONE]
    assert tokens, "expected streamed token chunks"
    assert "".join(c.text for c in tokens).strip()
    assert done
    assert done[0].usage is not None
    assert done[0].usage.total_tokens > 0


def test_orchestrator_run_records_metrics() -> None:
    before = _count_runs()
    orch = Orchestrator(provider=StubModelProvider())
    result = orch.run(build_hello_agent(), "ping")
    after = _count_runs()
    assert after == before + 1
    assert result.finish_reason.value == "stop"
    snap = get_metrics_store().snapshot()
    last = snap.recent[-1]
    assert last["succeeded"] is True
    assert last["model"] == "stub"


def test_orchestrator_run_wraps_retry_and_records_attempts(monkeypatch) -> None:
    # Force a single retry then success by making the stub fail once.
    calls = {"n": 0}

    class FlakyStub(StubModelProvider):
        def run(self, turn):
            calls["n"] += 1
            if calls["n"] == 1:
                raise __import__("chi_runtime.retry", fromlist=["ModelError"]).ModelError(
                    "blip", kind="server_error"
                )
            return super().run(turn)

    orch = Orchestrator(provider=FlakyStub())
    result = orch.run(build_hello_agent(), "retry me")
    # Recovered after one retry.
    assert result.finish_reason.value == "stop"
    snap = get_metrics_store().snapshot()
    last = snap.recent[-1]
    assert last["attempts"] == 2
    assert last["retried"] is True


def test_config_env_toggles_streaming(monkeypatch) -> None:
    monkeypatch.setenv("CHI_STREAM__ENABLED", "false")
    reload_config()
    assert get_config().streaming.enabled is False
    # Restore default for other tests.
    monkeypatch.delenv("CHI_STREAM__ENABLED", raising=False)
    reload_config()
