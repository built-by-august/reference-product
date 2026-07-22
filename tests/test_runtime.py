"""Unit tests for the agent runtime skeleton.

No network, no API keys — the default stub provider is deterministic and offline. A
fake provider also exercises the tool-calling loop so we prove that path works.
"""

from __future__ import annotations

from dataclasses import dataclass

from chi_runtime import (
    Agent,
    AgentResult,
    FinishReason,
    Orchestrator,
    StubModelProvider,
    ToolCall,
    ToolRegistry,
    Usage,
)
from chi_runtime.agents import build_hello_agent


@dataclass
class _FakeTurn:
    content: str
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: FinishReason
    model: str = "fake"


class EchoToolProvider:
    """Provider that, on the first turn, asks for the `echo` tool, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, turn):  # noqa: ANN001 - ModelTurn
        self.calls += 1
        if self.calls == 1:
            return _FakeTurn(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                usage=Usage(total_tokens=5, latency_ms=1.0),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        return _FakeTurn(
            content="Echoed 'hi' for you.",
            tool_calls=[],
            usage=Usage(total_tokens=3, latency_ms=1.0),
            finish_reason=FinishReason.STOP,
        )


def _make_agent_with_echo() -> Agent:
    tools = ToolRegistry()
    tools.register(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda a: {"echoed": a.get("text", "")},
    )
    return Agent(name="t", system_prompt="sys", tools=tools, model="stub")


def test_hello_agent_runs_with_stub_and_returns_result():
    agent = build_hello_agent()
    result = Orchestrator().run(agent, "What is Chimeric Intelligence?")
    assert isinstance(result, AgentResult)
    assert result.finish_reason == FinishReason.STOP
    assert "stub" in result.content.lower()
    assert result.trace_id
    assert "echo" in [t.name for t in agent.tools.all()]


def test_stub_acknowledges_prompt_text():
    agent = build_hello_agent()
    result = Orchestrator().run(agent, "ping")
    assert "ping" in result.content


def test_tool_calling_loop_executes_and_accumulates_usage():
    agent = _make_agent_with_echo()
    provider = EchoToolProvider()
    result = Orchestrator(provider=provider).run(agent, "echo hi")
    # Tool ran, loop continued, then a final STOP answer.
    assert result.finish_reason == FinishReason.STOP
    assert result.content == "Echoed 'hi' for you."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "echo"
    # Usage accumulates across both turns.
    assert result.usage.total_tokens == 8
    assert result.usage.latency_ms == 2.0


def test_max_turns_guardrail():
    class LoopProvider:
        def run(self, turn):  # noqa: ANN001
            return _FakeTurn(
                content="",
                tool_calls=[ToolCall(id="x", name="echo", arguments={"text": "x"})],
                usage=Usage(),
                finish_reason=FinishReason.TOOL_CALLS,
            )

    agent = _make_agent_with_echo()
    result = Orchestrator(LoopProvider()).run(agent, "loop")
    assert result.finish_reason == FinishReason.LENGTH


def test_stub_provider_is_injectable():
    orch = Orchestrator(StubModelProvider(model="stub-v2"))
    result = orch.run(build_hello_agent(), "hi")
    assert result.model == "stub-v2"
