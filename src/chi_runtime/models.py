"""Core data models for the Chimeric agent runtime.

These are the typed contracts that flow through the orchestrator: tools the model
can call, the messages in a conversation, and the result the runtime returns. Keeping
them in one small module makes the runtime easy to reason about and fully typed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class Role(str, Enum):
    """Message author roles, mirroring the common LLM chat convention."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Why a model turn stopped. `tool_calls` means the orchestrator should loop."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"


@dataclass
class Tool:
    """A tool the agent can call. `run` executes it with already-parsed arguments."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Any = None  # Callable[[dict], Any] — kept loosely typed to avoid import cycles


@dataclass
class ToolCall:
    """A single invocation the model wants to make."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A turn in the conversation. Tool results are attached with `tool_call_id`."""

    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name when role == TOOL

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Usage:
    """Observed cost/latency for a model turn. Zeroed out for the stub provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )

    @classmethod
    def zero(cls) -> Usage:
        return cls()


class StreamEventType(str, Enum):
    """Discriminator for a streaming chunk emitted by a provider."""

    TOKEN = "token"        # a piece of assistant text
    TOOL_CALL = "tool_call"  # a (possibly partial) tool call
    DONE = "done"          # stream finished
    ERROR = "error"        # stream aborted with an error


@dataclass
class StreamChunk:
    """One incremental event from a streaming model call.

    The harness streams these as tokens arrive; the web layer forwards them as
    Server-Sent Events. ``text`` is only meaningful on ``TOKEN`` chunks.
    """

    event: StreamEventType
    text: str = ""
    data: dict[str, Any] | None = None  # tool_call payload, error detail, etc.
    usage: Usage | None = None         # present on the DONE chunk
    finish_reason: FinishReason | None = None  # present on the DONE chunk
    trace_id: str | None = None          # present on the DONE chunk


@dataclass
class AgentResult:
    """The runtime's deliverable: the final assistant content plus full observability."""

    content: str
    finish_reason: FinishReason
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    trace_id: str = field(default_factory=_new_trace_id)
    model: str = "stub"
