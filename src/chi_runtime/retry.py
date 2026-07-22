"""Retry/backoff wrapper for model calls, with latency + attempt tracing.

Model calls are the flakiest, most expensive part of any agent product. This
module gives the runtime a single, observable seam for *trying again*:

* exponential backoff with a jitter and a hard cap (``RetryConfig``),
* a per-attempt record of latency and the error kind,
* a final ``RetryResult`` that reports how many attempts it took and how long
  the whole dance took — so the dashboard and metrics layer can show it.

The wrapper does NOT know anything about LLMs. It calls whatever
``callable`` you pass and inspects exceptions through a small ``error_kind``
hook. A ``ModelError`` helper lets providers raise structured, retry-classified
errors so the policy can decide what to retry.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .config import RetryConfig

T = TypeVar("T")

# Errors raised by model providers carry a ``kind`` so the retry policy can
# decide whether to retry (e.g. a 401 auth error must NOT be retried).
RETRYABLE_KINDS = ("timeout", "rate_limit", "server_error", "connection")


class ModelError(Exception):
    """A structured error from a model provider.

    ``kind`` should be one of the retryable kinds above (e.g. ``rate_limit``)
    or a non-retryable kind (e.g. ``auth``, ``invalid_request``).
    """

    def __init__(self, message: str, *, kind: str = "server_error", retryable: bool | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.retryable = RETRYABLE_KINDS.__contains__(kind) if retryable is None else retryable

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.kind}] {self.message}"


@dataclass
class AttemptRecord:
    """One try inside a retry sequence."""

    attempt: int
    latency_ms: float
    error_kind: str | None  # None on success
    error_message: str | None = None


@dataclass
class RetryResult:
    """Outcome of ``with_retry``: the value plus full traceability."""

    value: Any
    attempts: int
    total_latency_ms: float
    succeeded: bool
    error_kind: str | None = None
    error_message: str | None = None
    history: list[AttemptRecord] = field(default_factory=list)

    @property
    def retried(self) -> bool:
        return self.attempts > 1


def _sleep(delay: float) -> None:
    """Real sleep; injected seam kept simple. Tests override via monkeypatch."""
    time.sleep(delay)


def with_retry(
    fn: Callable[[], T],
    *,
    config: RetryConfig | None = None,
    error_kind: Callable[[Exception], str] | None = None,
    sleep: Callable[[float], None] = _sleep,
) -> RetryResult:
    """Run ``fn`` with exponential backoff. Returns a traceable ``RetryResult``.

    Parameters
    ----------
    fn:
        Zero-argument callable that performs the model call.
    config:
        ``RetryConfig`` policy. Defaults to built-in sane values.
    error_kind:
        Optional hook mapping an exception to a string kind. If omitted, we map
        a ``ModelError`` via its ``.kind`` and otherwise treat unknown exceptions
        as ``server_error`` (retryable by default policy).
    sleep:
        Injection seam for tests (so they don't actually wait).
    """
    cfg = config or RetryConfig()
    history: list[AttemptRecord] = []
    total_ms = 0.0
    last_kind: str | None = None
    last_msg: str | None = None

    for attempt in range(1, cfg.max_attempts + 1):
        start = time.perf_counter()
        try:
            value = fn()
            elapsed = (time.perf_counter() - start) * 1000.0
            total_ms += elapsed
            history.append(AttemptRecord(attempt=attempt, latency_ms=elapsed, error_kind=None))
            return RetryResult(
                value=value,
                attempts=attempt,
                total_latency_ms=total_ms,
                succeeded=True,
                history=history,
            )
        except Exception as exc:  # noqa: BLE001 - we re-raise, but want to record + decide
            elapsed = (time.perf_counter() - start) * 1000.0
            total_ms += elapsed
            kind = (error_kind(exc) if error_kind else _default_kind(exc))
            last_kind, last_msg = kind, str(exc)
            history.append(
                AttemptRecord(
                    attempt=attempt,
                    latency_ms=elapsed,
                    error_kind=kind,
                    error_message=str(exc),
                )
            )
            # Stop early on non-retryable errors.
            if not _is_retryable(kind, exc):
                return RetryResult(
                    value=None, attempts=attempt, total_latency_ms=total_ms,
                    succeeded=False, error_kind=kind, error_message=str(exc), history=history,
                )
            # Out of attempts?
            if attempt >= cfg.max_attempts:
                return RetryResult(
                    value=None, attempts=attempt, total_latency_ms=total_ms,
                    succeeded=False, error_kind=kind, error_message=str(exc), history=history,
                )
            # Exponential backoff with jitter, capped at max_delay.
            backoff = min(cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)), cfg.max_delay)
            jitter = backoff * 0.2 * random.random()
            sleep(backoff + jitter)

    # Should be unreachable (loop returns on last attempt), but be safe.
    return RetryResult(
        value=None, attempts=cfg.max_attempts, total_latency_ms=total_ms,
        succeeded=False, error_kind=last_kind, error_message=last_msg, history=history,
    )


def _default_kind(exc: Exception) -> str:
    if isinstance(exc, ModelError):
        return exc.kind
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "connection" in name or "connect" in name or "network" in name:
        return "connection"
    return "server_error"


def _is_retryable(kind: str, exc: Exception) -> bool:
    if isinstance(exc, ModelError):
        return exc.retryable
    return kind in RETRYABLE_KINDS
