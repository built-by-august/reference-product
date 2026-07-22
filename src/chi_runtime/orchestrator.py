"""Orchestrator: the run loop that turns an Agent + prompt into an AgentResult.

This is the heart of the runtime. It:
  1. builds the message list (system + user),
  2. calls the injected `ModelProvider`,
  3. if the model emits tool calls, executes them and loops (tool-calling scaffold),
  4. returns a fully-observed `AgentResult`.

The tool-calling branch is exercised by tests via a stub that returns a tool call, so the
loop is real, not dead code. With the default `StubModelProvider` it returns after one
turn and no tools fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .model import ModelProvider, StubModelProvider
from .models import (
    AgentResult,
    FinishReason,
    Message,
    Role,
    ToolCall,
    Usage,
)
from .tool import ToolRegistry

MAX_TURNS = 8  # guardrail so a misbehaving model can't loop forever


@dataclass
class Orchestrator:
    provider: ModelProvider = field(default_factory=StubModelProvider)

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
        total_usage = Usage()
        trace_tool_calls: list[ToolCall] = []

        for _ in range(max_turns):
            turn = self.provider.run(
                _turn(messages, tools, agent.model, agent.temperature)
            )
            # Accumulate observed cost/latency across turns.
            total_usage.prompt_tokens += turn.usage.prompt_tokens
            total_usage.completion_tokens += turn.usage.completion_tokens
            total_usage.total_tokens += turn.usage.total_tokens
            total_usage.latency_ms += turn.usage.latency_ms

            content = getattr(turn, "content", "")
            tool_calls = list(getattr(turn, "tool_calls", []) or [])

            assistant = Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
            messages.append(assistant)

            if not tool_calls or turn.finish_reason != FinishReason.TOOL_CALLS:
                return AgentResult(
                    content=content,
                    finish_reason=FinishReason(turn.finish_reason),
                    messages=messages,
                    tool_calls=trace_tool_calls,
                    usage=total_usage,
                    model=turn.model,
                )

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

        return AgentResult(
            content="[orchestrator] reached max turns without finishing.",
            finish_reason=FinishReason.LENGTH,
            messages=messages,
            tool_calls=trace_tool_calls,
            usage=total_usage,
            model=getattr(turn, "model", agent.model) if "turn" in dir() else agent.model,
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
