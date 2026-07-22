"""Model provider protocol + a deterministic stub.

The runtime talks to models only through `ModelProvider.run(turn)`. Today the default
`StubModelProvider` returns a canned response so the whole system runs offline with
zero cost. To wire in a real frontier model, implement this protocol (e.g. an
OpenAI/Anthropic-backed provider) and inject it into the orchestrator — nothing else
changes. Cost and latency are captured on every `Usage` so they are observable by default.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .models import FinishReason, Message, StreamChunk, StreamEventType, Tool, Usage, _new_trace_id


@dataclass
class ModelTurn:
    """Everything a provider needs to produce one assistant turn."""

    messages: list[Message]
    tools: list[Tool]
    model: str
    temperature: float = 0.0


class ModelProvider(Protocol):
    """The single seam between the orchestrator and any model backend."""

    def run(self, turn: ModelTurn) -> Any:
        """Return an object with `.content`, `.tool_calls`, `.usage`, `.finish_reason`."""
        ...

    def run_stream(self, turn: ModelTurn) -> Iterator[StreamChunk]:
        """Yield incremental chunks as the model generates.

        Providers that do not support streaming may raise ``NotImplementedError``;
        the orchestrator/harness will fall back to a buffered ``run`` result.
        """
        ...


@dataclass
class StubResponse:
    content: str
    tool_calls: list[Any]
    usage: Usage
    finish_reason: FinishReason
    model: str


class StubModelProvider:
    """Deterministic, offline model stand-in.

    Echoes acknowledgement of the last user prompt so the harness always returns a real,
    inspectable result without network or API keys.
    """

    def __init__(self, model: str = "stub") -> None:
        self.model = model

    def run(self, turn: ModelTurn) -> StubResponse:
        start = time.perf_counter()
        last_user = next(
            (m for m in reversed(turn.messages) if m.role.value == "user"), None
        )
        prompt = last_user.content if last_user else "(no prompt)"
        content = (
            f"[stub] Received prompt: {prompt!r}. "
            f"No model is configured yet — wire a ModelProvider to answer for real. "
            f"Tools available: {[t.name for t in turn.tools]}."
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return StubResponse(
            content=content,
            tool_calls=[],
            usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=elapsed_ms,
        ),
            finish_reason=FinishReason.STOP,
            model=self.model,
        )

    def run_stream(self, turn: ModelTurn) -> Iterator[StreamChunk]:
        """Stream the canned acknowledgment word-by-word so the harness can show tokens.

        Mirrors ``run`` but yields a ``TOKEN`` chunk per whitespace-separated word,
        then a final ``DONE`` chunk carrying the aggregate ``Usage``.
        """
        start = time.perf_counter()
        last_user = next(
            (m for m in reversed(turn.messages) if m.role.value == "user"), None
        )
        prompt = last_user.content if last_user else "(no prompt)"
        words = (
            f"[stub] Received prompt: {prompt!r}. "
            f"No model is configured yet — wire a ModelProvider to answer for real. "
            f"Tools available: {[t.name for t in turn.tools]}."
        ).split(" ")
        for w in words:
            yield StreamChunk(event=StreamEventType.TOKEN, text=w + " ")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        yield StreamChunk(
            event=StreamEventType.DONE,
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(words),
                total_tokens=len(words),
                latency_ms=elapsed_ms,
            ),
            finish_reason=FinishReason.STOP,
            trace_id=_new_trace_id(),
        )
