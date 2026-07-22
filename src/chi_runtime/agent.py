"""Agent definition.

An Agent is a declarative spec: a name, a system prompt, and the set of tools it may
use. The orchestrator turns an Agent + a user prompt into an `AgentResult`. Keeping the
agent as data (not behavior) makes it easy to register many reference agents and inspect
their shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool import ToolRegistry


@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    model: str = "stub"
    temperature: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "tools": [t.name for t in self.tools.all()],
        }
