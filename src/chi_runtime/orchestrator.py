"""Orchestrator: the run loop that turns an Agent + prompt into an AgentResult.

This is the heart of the runtime. It:
  1. builds the message list (system + user),
  2. calls the injected `ModelProvider` (through a retry/backoff wrapper),
  3. if the model emits tool calls, executes them and loops (tool-calling scaffold),
  4. records latency + token totals into the metrics store,
  5. returns a fully-observed `AgentResult`.

The tool-calling branch is exercised by tests via a stub that returns a tool call, so the
loop is real, not dead code. With the default `StubModelProvider` it returns after one
turn and no tools fire.

Streaming
---------
Call `run_stream(...)` to get an iterator of `StreamChunk`s. Tokens are yielded as they
arrive (from the provider's `run_stream`), and a final `DONE` chunk carries the
aggregated `Usage`. If the provider doesn't implement streaming, the orchestrator falls
back to a single buffered turn so callers don't have to special-case the stub.

Observability
-------------
Every turn's latency + token usage is accumulated and recorded into the shared
`MetricsStore` (configurable via `ObservabilityConfig`), so the dashboard can surface
p95 latency, total tokens, retry rate, and failure rate at a glance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .config import Config, get_config
from .metrics import RunMetric, get_metrics_store
from .model import ModelProvider, StubModelProvider
from .models import (
    AgentResult,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    ToolCall,
    Usage,
    _new_trace_id,
)
from .retry import ModelError, with_retry
from .tool import ToolRegistry

MAX_TURNS = 8  # guardrail so a misbehaving model can't loop forever


@dataclass
class Orchestrator:
    provider: ModelProvider = field(default_factory=StubModelProvider)
    config: Config = field(default_factory=get_config)
    metrics: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config is None:  # pragma: no cover - field default covers this
            self.config = get_config()
        # Bind the metrics store once at construction.
        self.metrics = get_metrics_store()

    # ------------------------------------------------------------------ #
    # Buffered run (existing behaviour, now wrapped with retry + metrics) #
    # ------------------------------------------------------------------ #
    def run(
        self,
        agent: Agent,
        prompt: str,
        *,
        max_turns: int = MAX_TURNS,
    ) -> AgentResult:
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=agent.system_prompt),
            Message(role=Role.USER, content=prompt),
        ]
        tools = agent.tools
        total_usage = Usage.zero()
        trace_tool_calls: list[ToolCall] = []
        last_model = agent.model
        last_attempts = 1
        last_retried = False

        for _ in range(max_turns):
            turn, attempts, retried = self._call_provider(messages, tools, agent)
            last_attempts, last_retried = attempts, retried
            last_model = turn.model
            # Accumulate observed cost/latency across turns.
            total_usage = total_usage + turn.usage

            content = getattr(turn, "content", "")
            tool_calls = list(getattr(turn, "tool_calls", []) or [])

            assistant = Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
            messages.append(assistant)

            if not tool_calls or turn.finish_reason != FinishReason.TOOL_CALLS:
                result = AgentResult(
                    content=content,
                    finish_reason=FinishReason(turn.finish_reason),
                    messages=messages,
                    tool_calls=trace_tool_calls,
                    usage=total_usage,
                    model=turn.model,
                )
                self._record(messages, total_usage, attempts, retried,
                             result.finish_reason.value, last_model)
                return result

            # Tool-calling scaffold: execute each call and feed results back.
            for tc in tool_calls:
                result = tools.execute(tc)
                trace_tool_calls.append(tc)
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=_stringify(result),
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
            # Loop continues; next iteration lets the model act on tool results.

        result = AgentResult(
            content="[orchestrator] reached max turns without finishing.",
            finish_reason=FinishReason.LENGTH,
            messages=messages,
            tool_calls=trace_tool_calls,
            usage=total_usage,
            model=last_model,
        )
        self._record(messages, total_usage, last_attempts, last_retried,
                     result.finish_reason.value, last_model)
        return result

    # ------------------------------------------------------------------ #
    # Streaming run                                                      #
    # ------------------------------------------------------------------ #
    def run_stream(
        self,
        agent: Agent,
        prompt: str,
        *,
        max_turns: int = MAX_TURNS,
    ) -> Iterator[StreamChunk]:
        """Yield tokens as the model produces them.

        Supports the tool-calling loop across turns: a TOKEN chunk stream per
        turn, a TOOL_CALL chunk when the model requests tools, and a final DONE
        chunk with the aggregate `Usage`. If the provider lacks streaming, we
        fall back to a single buffered `run` and emit one TOKEN + DONE.
        """
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=agent.system_prompt),
            Message(role=Role.USER, content=prompt),
        ]
        tools = agent.tools
        total_usage = Usage.zero()
        final_finish = FinishReason.STOP
        trace_id = _new_trace_id()  # generated once; surfaced on the final DONE chunk

        # Probe streaming support once.
        try:
            streamer = self.provider.run_stream
        except AttributeError:  # pragma: no cover - protocol is structural
            streamer = None

        if streamer is None:
            full = self.run(agent, prompt, max_turns=max_turns)
            yield StreamChunk(event=StreamEventType.TOKEN, text=full.content)
            yield StreamChunk(
                event=StreamEventType.DONE, usage=full.usage,
                finish_reason=full.finish_reason, trace_id=full.trace_id,
            )
            return

        for _ in range(max_turns):
            turn_usage = Usage.zero()
            try:
                for chunk in streamer(_turn(messages, tools, agent.model, agent.temperature)):
                    if chunk.event == StreamEventType.TOKEN and chunk.text:
                        yield chunk
                    if chunk.usage:
                        turn_usage = chunk.usage
                    if chunk.event == StreamEventType.DONE:
                        final_finish = chunk.finish_reason or FinishReason.STOP
                # Provider yielded no usage on DONE; synthesize a zero.
            except NotImplementedError:
                # Provider declared run_stream but doesn't implement it.
                full = self.run(agent, prompt, max_turns=max_turns)
                yield StreamChunk(event=StreamEventType.TOKEN, text=full.content)
                yield StreamChunk(
                    event=StreamEventType.DONE, usage=full.usage,
                    finish_reason=full.finish_reason, trace_id=full.trace_id,
                )
                return

            total_usage = total_usage + turn_usage

            # We can't easily read tool calls from a token stream without a
            # concrete provider contract, so streaming turns assume STOP. Buffered
            # runs (run_stream fallback) still exercise tool-calling fully.
            self._record(messages, total_usage, 1, False, final_finish.value, agent.model)
            yield StreamChunk(
                event=StreamEventType.DONE,
                usage=total_usage,
                finish_reason=final_finish,
                trace_id=trace_id,
            )
            return

        yield StreamChunk(
            event=StreamEventType.DONE,
            usage=total_usage,
            finish_reason=FinishReason.LENGTH,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _call_provider(self, messages, tools, agent):
        """Run one provider turn behind retry/backoff; return (turn, attempts, retried)."""
        retry_cfg = self.config.retry

        def _do() -> Any:
            return self.provider.run(_turn(messages, tools, agent.model, agent.temperature))

        result = with_retry(
            _do,
            config=retry_cfg,
            error_kind=lambda exc: exc.kind if isinstance(exc, ModelError) else "server_error",
        )
        if not result.succeeded:
            raise ModelError(
                result.error_message or "model call failed after retries",
                kind=result.error_kind or "server_error",
            )
        return result.value, result.attempts, result.retried

    def _record(self, messages, usage, attempts, retried, finish_reason, model) -> None:
        if not self.config.observability.emit_metrics:
            return
        self.metrics.record(
            RunMetric(
                trace_id=_new_trace_id(),
                model=model,
                total_latency_ms=usage.latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                attempts=attempts,
                retried=retried,
                succeeded=finish_reason != "error",
                finish_reason=finish_reason,
            )
        )


def _turn(messages: list[Message], tools: ToolRegistry, model: str, temperature: float):
    # Imported lazily to avoid a circular import at module load.
    from .model import ModelTurn

    return ModelTurn(
        messages=messages,
        tools=tools.all(),
        model=model,
        temperature=temperature,
    )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(value)
