"""Offline tests for the real frontier ModelProviders (CHI-9 / CHI-1.2).

These exercise the OpenAI and Anthropic providers against ``httpx.MockTransport``
so the real-request shaping + Usage accounting is proven without network or API
keys. The orchestrator is run end-to-end with a mocked provider to confirm token
usage flows into the AgentResult the same way the stub does, and that the streaming
fallback (providers that don't implement ``run_stream`` yet) still returns content.
"""

from __future__ import annotations

import json

import httpx
import pytest

from chi_runtime import Agent, Orchestrator, Role, ToolRegistry, build_provider
from chi_runtime.models import FinishReason, Message, ToolCall
from chi_runtime.providers import AnthropicModelProvider, OpenAIModelProvider

_KEY = "test-key"


def _openai_mock_handler(expected_model: str):
    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content.decode())
        assert sent["model"] == expected_model
        assert sent["messages"][-1]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenAI."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            },
        )

    return handler


def _anthropic_mock_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-5-haiku-latest",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Anthropic."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        )

    return handler


def _mock_openai(model: str) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_openai_mock_handler(model)))


def _mock_anthropic() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_anthropic_mock_handler()))


def _make_agent() -> Agent:
    tools = ToolRegistry()
    tools.register(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda a: {"echoed": a.get("text", "")},
    )
    return Agent(name="hello", system_prompt="sys", tools=tools, model="stub")


def _turn(role_text, model: str = "stub"):
    from chi_runtime.model import ModelTurn

    return ModelTurn(
        messages=[Message(role=Role(value=r), content=t) for r, t in role_text],
        tools=[],
        model=model,
        temperature=0.0,
    )


def test_openai_provider_returns_real_content_with_usage():
    provider = OpenAIModelProvider(api_key=_KEY, model="gpt-4o-mini")
    provider._client = _mock_openai("gpt-4o-mini")
    resp = provider.run(_turn([("user", "Hi")], model="stub"))
    assert resp.content == "Hello from OpenAI."
    assert resp.finish_reason == FinishReason.STOP
    assert resp.usage.prompt_tokens == 12
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 15
    assert resp.usage.latency_ms >= 0
    assert resp.model == "gpt-4o-mini"
    provider.close()


def test_anthropic_provider_returns_real_content_with_usage():
    provider = AnthropicModelProvider(api_key=_KEY)
    provider._client = _mock_anthropic()
    resp = provider.run(_turn([("user", "Hi")], model="stub"))
    assert resp.content == "Hello from Anthropic."
    assert resp.finish_reason == FinishReason.STOP
    assert resp.usage.prompt_tokens == 20
    assert resp.usage.completion_tokens == 4
    assert resp.usage.total_tokens == 24
    provider.close()


def test_openai_provider_normalizes_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text": "hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    provider = OpenAIModelProvider(api_key=_KEY, model="gpt-4o-mini")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    resp = provider.run(_turn([("user", "echo hi")], model="stub"))
    assert resp.finish_reason == FinishReason.TOOL_CALLS
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.name == "echo"
    assert tc.arguments == {"text": "hi"}
    provider.close()


def test_build_provider_factory_selects_backends():
    assert build_provider("stub").__class__.__name__ == "StubModelProvider"
    assert build_provider("openai", api_key=_KEY).__class__.__name__ == "OpenAIModelProvider"
    assert build_provider("anthropic", api_key=_KEY).__class__.__name__ == "AnthropicModelProvider"


def test_build_provider_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_provider("nope")


def test_openai_provider_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIModelProvider()


def test_orchestrator_runs_real_provider_end_to_end():
    """A mocked real provider is indistinguishable from the stub to the orchestrator."""
    provider = OpenAIModelProvider(api_key=_KEY, model="gpt-4o-mini")
    provider._client = _mock_openai("gpt-4o-mini")
    agent = _make_agent()
    result = Orchestrator(provider=provider).run(agent, "What is Chimeric Intelligence?")
    assert result.content == "Hello from OpenAI."
    assert result.usage.total_tokens == 15
    assert result.finish_reason == FinishReason.STOP
    # Observability surfaced exactly like the stub path.
    assert result.trace_id
    provider.close()


def test_stub_model_name_is_ignored_by_real_provider():
    """Agents default model to 'stub'; a real provider must use its own default."""
    provider = OpenAIModelProvider(api_key=_KEY)  # no explicit model
    provider._client = _mock_openai("gpt-4o-mini")
    resp = provider.run(_turn([("user", "hi")], model="stub"))
    assert resp.model == "gpt-4o-mini"
    provider.close()


def test_real_provider_streaming_falls_back_to_buffered_run():
    """Orchestrator.run_stream must still yield content when a provider lacks run_stream."""
    provider = AnthropicModelProvider(api_key=_KEY)
    provider._client = _mock_anthropic()
    orch = Orchestrator(provider=provider)
    chunks = list(orch.run_stream(_make_agent(), "hi"))
    token_chunks = [c for c in chunks if c.event.value == "token"]
    done_chunks = [c for c in chunks if c.event.value == "done"]
    assert token_chunks  # at least one TOKEN chunk
    assert "".join(c.text for c in token_chunks) == "Hello from Anthropic."
    assert done_chunks
    assert done_chunks[0].usage is not None
    provider.close()
