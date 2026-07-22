"""Real frontier model providers behind the ModelProvider protocol.

Two providers are included. Both talk to the vendor REST APIs over ``httpx``
(already a runtime dependency), so wiring a real model adds **no new packages**:

* ``OpenAIModelProvider``    -> OpenAI Chat Completions API
* ``AnthropicModelProvider`` -> Anthropic Messages API

Both implement the ``ModelProvider`` protocol (``run`` + ``run_stream``) and return
the same shaped response the orchestrator already understands (``.content``,
``.tool_calls``, ``.usage``, ``.finish_reason``, ``.model``). Swapping providers is
therefore a one-line change in the config/CLI — nothing else in the runtime needs to
know which backend is live.

API keys are resolved from, in order: (1) an explicit ``api_key=`` argument,
(2) the harness ``ModelConfig.api_key_env`` (default ``CHI_API_KEY``), then
(3) the vendor's own env var (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) as a
convenience fallback. ``build_provider(mode, ...)`` is the single seam the CLI and
web layer use to select a backend without touching the orchestrator or agents.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .model import ModelProvider, ModelTurn, StubResponse
from .models import FinishReason, Message, StreamChunk, Tool, ToolCall, Usage


def _resolve_model(turn_model: str, default_model: str) -> str:
    """Pick the model to call.

    Agents default their ``model`` field to ``"stub"``. A real provider ignores that
    placeholder and uses its configured default unless the agent/config names an
    explicit, real model (e.g. ``gpt-4o-mini``).
    """
    if turn_model and turn_model not in ("stub", ""):
        return turn_model
    return default_model


def _safe_json(value: Any) -> dict[str, Any]:
    """Parse a JSON tool-call argument string, tolerating empties/malformed input."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"__raw__": value}
    except (json.JSONDecodeError, TypeError):
        return {"__raw__": value}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    # Some APIs return content as a list of text parts; join the text ones.
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


@dataclass
class _Config:
    """Shared knobs for the real providers."""

    api_key: str
    model: str
    base_url: str
    timeout: float


class OpenAIModelProvider:
    """Talks to the OpenAI Chat Completions API over a reusable ``httpx`` client."""

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
    VENDOR_ENV = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        api_key_env: str = "CHI_API_KEY",
    ) -> None:
        key = self._resolve_key(api_key, api_key_env)
        self.config = _Config(
            api_key=key,
            model=model or os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL),
            timeout=timeout,
        )
        self._client = httpx.Client(timeout=timeout)

    @staticmethod
    def _resolve_key(api_key: str | None, api_key_env: str) -> str:
        key = api_key or os.environ.get(api_key_env, "")
        if not key:
            key = os.environ.get(OpenAIModelProvider.VENDOR_ENV, "")
        if not key:
            raise ValueError(
                "OpenAIModelProvider requires an API key: set "
                f"{api_key_env} (or OPENAI_API_KEY) or pass api_key=."
            )
        return key

    # --- request shaping --------------------------------------------------
    @staticmethod
    def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role.value == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": m.content,
                        "tool_call_id": m.tool_call_id or "",
                    }
                )
                continue
            entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        return out

    @staticmethod
    def _to_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _body(self, turn: ModelTurn) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": _resolve_model(turn.model, self.config.model),
            "messages": self._to_messages(turn.messages),
            "temperature": turn.temperature,
        }
        tools = self._to_tools(turn.tools)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    # --- response normalization ------------------------------------------
    def _normalize(self, data: dict[str, Any], elapsed_ms: float) -> StubResponse:
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=_safe_json(tc["function"]["arguments"]),
            )
            for tc in msg.get("tool_calls", []) or []
        ]
        finish_map = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
        }
        finish_reason = finish_map.get(choice.get("finish_reason", "stop"), FinishReason.STOP)
        u = data.get("usage", {})
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
            latency_ms=elapsed_ms,
        )
        return StubResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            model=data.get("model", self.config.model),
        )

    def run(self, turn: ModelTurn) -> StubResponse:
        start = time.perf_counter()
        resp = self._client.post(
            self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=self._body(turn),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        return self._normalize(resp.json(), elapsed_ms)

    def run_stream(self, turn: ModelTurn) -> Iterator[StreamChunk]:
        """Real token streaming is not implemented for the buffered OpenAI call yet.

        The orchestrator catches ``NotImplementedError`` and falls back to a single
        buffered ``run`` result, so the harness/UI keep working without special-casing.
        """
        raise NotImplementedError("OpenAIModelProvider.run_stream is not implemented yet.")

    def close(self) -> None:
        self._client.close()


class AnthropicModelProvider:
    """Talks to the Anthropic Messages API over a reusable ``httpx`` client."""

    DEFAULT_MODEL = "claude-3-5-haiku-latest"
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MAX_TOKENS = 1024
    VENDOR_ENV = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 60.0,
        api_key_env: str = "CHI_API_KEY",
    ) -> None:
        key = self._resolve_key(api_key, api_key_env)
        self.config = _Config(
            api_key=key,
            model=model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL),
            base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL", self.DEFAULT_BASE_URL),
            timeout=timeout,
        )
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    @staticmethod
    def _resolve_key(api_key: str | None, api_key_env: str) -> str:
        key = api_key or os.environ.get(api_key_env, "")
        if not key:
            key = os.environ.get(AnthropicModelProvider.VENDOR_ENV, "")
        if not key:
            raise ValueError(
                "AnthropicModelProvider requires an API key: set "
                f"{api_key_env} (or ANTHROPIC_API_KEY) or pass api_key=."
            )
        return key

    # --- request shaping --------------------------------------------------
    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        system_parts: list[str] = []
        rest: list[Message] = []
        for m in messages:
            if m.role.value == "system":
                system_parts.append(m.content)
            else:
                rest.append(m)
        return "\n\n".join(system_parts), rest

    @staticmethod
    def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role.value == "assistant":
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": content})
            elif m.role.value == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "",
                                "content": m.content,
                            }
                        ],
                    }
                )
            else:  # user
                out.append({"role": "user", "content": m.content})
        return out

    @staticmethod
    def _to_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _body(self, turn: ModelTurn) -> dict[str, Any]:
        system, rest = self._split_system(turn.messages)
        body: dict[str, Any] = {
            "model": _resolve_model(turn.model, self.config.model),
            "messages": self._to_messages(rest),
            "max_tokens": self.max_tokens,
            "temperature": turn.temperature,
        }
        if system:
            body["system"] = system
        tools = self._to_tools(turn.tools)
        if tools:
            body["tools"] = tools
        return body

    # --- response normalization ------------------------------------------
    def _normalize(self, data: dict[str, Any], elapsed_ms: float) -> StubResponse:
        content = _content_to_text(
            [b for b in data.get("content", []) if b.get("type") == "text"]
        )
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}) or {},
                    )
                )
        stop_map = {
            "end_turn": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "max_tokens": FinishReason.LENGTH,
        }
        finish_reason = stop_map.get(data.get("stop_reason", "end_turn"), FinishReason.STOP)
        u = data.get("usage", {})
        usage = Usage(
            prompt_tokens=u.get("input_tokens", 0),
            completion_tokens=u.get("output_tokens", 0),
            total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
            latency_ms=elapsed_ms,
        )
        return StubResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            model=data.get("model", self.config.model),
        )

    def run(self, turn: ModelTurn) -> StubResponse:
        start = time.perf_counter()
        resp = self._client.post(
            self.config.base_url,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=self._body(turn),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        return self._normalize(resp.json(), elapsed_ms)

    def run_stream(self, turn: ModelTurn) -> Iterator[StreamChunk]:
        """Real token streaming is not implemented for the buffered Anthropic call yet.

        The orchestrator catches ``NotImplementedError`` and falls back to a single
        buffered ``run`` result.
        """
        raise NotImplementedError("AnthropicModelProvider.run_stream is not implemented yet.")

    def close(self) -> None:
        self._client.close()


def build_provider(
    mode: str = "stub",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "CHI_API_KEY",
    timeout: float = 60.0,
) -> ModelProvider:
    """Construct a ModelProvider from a mode string.

    ``mode`` is one of ``stub`` | ``openai`` | ``anthropic``. API keys fall back to
    ``api_key_env`` (default ``CHI_API_KEY``) then the vendor env var. Returned object
    satisfies the ``ModelProvider`` protocol the orchestrator depends on.
    """
    mode = (mode or "stub").lower()
    # Resolve the API key once: explicit arg wins, otherwise read from api_key_env
    # (default CHI_API_KEY) so callers can point at a non-vendor env var.
    resolved_key = api_key if api_key is not None else os.environ.get(api_key_env)
    if mode == "stub":
        from .model import StubModelProvider

        return StubModelProvider(model=model or "stub")
    if mode == "openai":
        return OpenAIModelProvider(
            api_key=resolved_key, model=model, base_url=base_url, timeout=timeout
        )
    if mode == "anthropic":
        return AnthropicModelProvider(
            api_key=resolved_key, model=model, base_url=base_url, timeout=timeout
        )
    raise ValueError(
        f"Unknown provider mode: {mode!r} (expected one of stub|openai|anthropic)."
    )
