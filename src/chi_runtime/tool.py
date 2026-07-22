"""Tool scaffold + a tiny registry.

A tool is a callable that takes parsed JSON arguments and returns a JSON-serializable
result. The registry lets agents and the orchestrator look tools up by name when the
model emits tool calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import Tool

ToolFn = Callable[[dict[str, Any]], Any]


@dataclass
class ToolRegistry:
    """Maps tool names to Tool definitions. Thread-safe enough for single-threaded runs."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: ToolFn,
    ) -> Tool:
        tool = Tool(name=name, description=description, parameters=parameters, run=fn)
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, call: Any) -> Any:
        """Execute a ToolCall-like object by looking it up in the registry."""
        tool = self._tools.get(call.name)
        if tool is None or tool.run is None:
            return {"error": f"unknown or unexecutable tool: {call.name}"}
        return tool.run(call.arguments)
