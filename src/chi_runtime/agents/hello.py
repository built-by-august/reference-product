"""Reference "hello agent".

This is the canonical runnable agent: accepts a prompt, runs it through the orchestrator
with the (stub) model, and returns a result. It registers one demonstration tool
(`echo`) so the tool-calling scaffold is real and testable, even though the stub model
does not invoke it by default.
"""

from __future__ import annotations

from ..agent import Agent
from ..tool import ToolRegistry


def build_hello_agent() -> Agent:
    """Construct the reference hello agent with its tool set."""
    tools = ToolRegistry()
    tools.register(
        name="echo",
        description="Echo back the provided text. Useful as a trivial tool-calling demo.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=lambda args: {"echoed": args.get("text", "")},
    )
    return Agent(
        name="hello",
        system_prompt=(
            "You are the Chimeric reference 'hello agent'. You acknowledge the user's "
            "prompt and explain that you are a stubbed agent until a real model provider "
            "is wired in. If asked to echo something, use the echo tool."
        ),
        tools=tools,
        model="stub",
    )
