"""In-process metrics store for latency and token tracing.

The dashboard ("observability hardening") needs real numbers: how many runs,
what the latency and token distribution looks like, how often we retried, and
how often calls failed. This module is the single place that records and
aggregates per-run observations.

It is intentionally dependency-free and safe to use from multiple threads
(each mutation takes a lock). A ring buffer caps memory so a long-running
harness can't grow it unbounded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunMetric:
    """One observed model-call run."""

    trace_id: str
    model: str
    total_latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 1
    retried: bool = False
    succeeded: bool = True
    error_kind: str | None = None
    finish_reason: str = "stop"
    ts: float = field(default_factory=time.time)


@dataclass
class MetricsSnapshot:
    """Aggregated view of all recorded runs (what the dashboard consumes)."""

    total_runs: int
    succeeded_runs: int
    failed_runs: int
    retried_runs: int
    total_tokens: int
    total_latency_ms: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_attempts: float
    recent: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsStore:
    """Thread-safe ring-buffer of run metrics with rolling aggregates."""

    def __init__(self, max_runs: int = 100) -> None:
        self._max_runs = 100 if max_runs < 1 else max_runs
        self._runs: list[RunMetric] = []
        self._lock = threading.Lock()

    def record(self, run: RunMetric) -> None:
        with self._lock:
            self._runs.append(run)
            if len(self._runs) > self._max_runs:
                # Drop oldest to keep a bounded history.
                self._runs = self._runs[-self._max_runs :]

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            runs = list(self._runs)
        total = len(runs)
        if total == 0:
            return MetricsSnapshot(
                total_runs=0, succeeded_runs=0, failed_runs=0, retried_runs=0,
                total_tokens=0, total_latency_ms=0.0, avg_latency_ms=0.0,
                p95_latency_ms=0.0, avg_attempts=0.0, recent=[],
            )
        latencies = sorted(r.total_latency_ms for r in runs)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))]
        return MetricsSnapshot(
            total_runs=total,
            succeeded_runs=sum(1 for r in runs if r.succeeded),
            failed_runs=sum(1 for r in runs if not r.succeeded),
            retried_runs=sum(1 for r in runs if r.retried),
            total_tokens=sum(r.total_tokens for r in runs),
            total_latency_ms=sum(r.total_latency_ms for r in runs),
            avg_latency_ms=sum(r.total_latency_ms for r in runs) / total,
            p95_latency_ms=p95,
            avg_attempts=sum(r.attempts for r in runs) / total,
            recent=[asdict(r) for r in runs[-25:]],
        )

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


# A process-wide store used by the runtime + web layer unless they inject their own.
_default_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    return _default_store
