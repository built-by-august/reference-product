"""Tests for CHI-1.3: retry/backoff and the in-process metrics store."""

from __future__ import annotations

from chi_runtime.config import RetryConfig
from chi_runtime.metrics import MetricsStore, RunMetric
from chi_runtime.retry import ModelError, with_retry


def test_success_first_try_no_retry() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    res = with_retry(fn, sleep=lambda _s: None)
    assert res.succeeded
    assert res.attempts == 1
    assert res.retried is False
    assert res.value == "ok"
    assert calls["n"] == 1


def test_retries_until_success_and_records_attempts() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ModelError("rate limited", kind="rate_limit")
        return "recovered"

    res = with_retry(flaky, config=RetryConfig(max_attempts=5), sleep=lambda _s: None)
    assert res.succeeded
    assert res.attempts == 3
    assert res.retried is True
    assert res.value == "recovered"
    # History records the two failures then the success.
    assert res.history[0].error_kind == "rate_limit"
    assert res.history[-1].error_kind is None


def test_non_retryable_fails_fast() -> None:
    calls = {"n": 0}

    def auth_fail() -> str:
        calls["n"] += 1
        raise ModelError("bad key", kind="auth", retryable=False)

    res = with_retry(auth_fail, sleep=lambda _s: None)
    assert res.succeeded is False
    assert res.attempts == 1  # did not retry an auth error
    assert res.error_kind == "auth"
    assert calls["n"] == 1


def test_exhausts_attempts_then_reports_failure() -> None:
    def always() -> str:
        raise ModelError("down", kind="server_error")

    res = with_retry(always, config=RetryConfig(max_attempts=3), sleep=lambda _s: None)
    assert res.succeeded is False
    assert res.attempts == 3
    assert res.error_kind == "server_error"


def test_metrics_store_aggregates() -> None:
    store = MetricsStore(max_runs=10)
    store.record(RunMetric(trace_id="a", model="stub", total_latency_ms=100.0,
                           total_tokens=10, attempts=1, retried=False, succeeded=True))
    store.record(RunMetric(trace_id="b", model="stub", total_latency_ms=200.0,
                           total_tokens=20, attempts=3, retried=True, succeeded=True))
    store.record(RunMetric(trace_id="c", model="stub", total_latency_ms=50.0,
                           total_tokens=5, attempts=2, retried=True, succeeded=False,
                           error_kind="server_error"))

    snap = store.snapshot()
    assert snap.total_runs == 3
    assert snap.succeeded_runs == 2
    assert snap.failed_runs == 1
    assert snap.retried_runs == 2
    assert snap.total_tokens == 35
    assert snap.total_latency_ms == 350.0
    assert snap.avg_latency_ms == 350.0 / 3
    # p95 with sorted [50, 100, 200] -> index ~1 -> 100.
    assert snap.p95_latency_ms == 100.0
    assert snap.recent  # recent runs present


def test_metrics_store_cap() -> None:
    store = MetricsStore(max_runs=3)
    for i in range(10):
        store.record(RunMetric(trace_id=str(i), model="stub", total_latency_ms=float(i)))
    assert len(store.snapshot().recent) <= 3
    assert store.snapshot().total_runs == 3
