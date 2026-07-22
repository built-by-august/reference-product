"""Lightweight B2C demo web service for the Chimeric reference product.

This is the thin web layer (CHI-1.5) that sits in front of the agent runtime
skeleton. It imports the already-built ``chi_runtime`` orchestrator and the
reference ``hello`` agent, exposes a tiny JSON API, and serves a single-page
visual-first UI from ``static/``.

Design notes
------------
* No new agent logic lives here — the runtime is the source of truth, this
  service only calls ``Orchestrator.run(agent, prompt)`` and renders the
  ``AgentResult`` (content + observability) back to the browser.
* The same ``Orchestrator`` is shared across requests. Swapping the stub model
  for a real frontier provider is a one-line change in the runtime and is
  picked up here automatically — cost/latency keep flowing through.
* Model and tool wiring are read from the agent at startup so the UI can show
  what the agent is capable of without hard-coding anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from chi_runtime import Orchestrator
from chi_runtime.agents import build_hello_agent
from chi_runtime.models import AgentResult, FinishReason, Role, Usage

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Allow overriding the UI directory (handy for local iteration without a rebuild).
STATIC_DIR = Path(os.environ.get("CHI_UI_DIR", str(STATIC_DIR)))

app = FastAPI(title="Chimeric B2C Demo", version="0.1.0")

# One orchestrator + agent for the process. Cheap to build and reused per request.
orchestrator = Orchestrator()
agent = build_hello_agent()


def _result_to_dict(result: AgentResult) -> dict[str, Any]:
    """Serialize an AgentResult into a JSON-friendly shape for the UI."""
    usage: Usage = result.usage
    return {
        "content": result.content,
        "model": result.model,
        "finish_reason": result.finish_reason.value
        if isinstance(result.finish_reason, FinishReason)
        else str(result.finish_reason),
        "trace_id": result.trace_id,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in result.tool_calls
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": round(usage.latency_ms, 2),
        },
        "messages": [
            {
                "role": m.role.value if isinstance(m.role, Role) else str(m.role),
                "content": m.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in m.tool_calls
                ]
                or None,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
            for m in result.messages
        ],
    }


def _agent_info() -> dict[str, Any]:
    return {
        "name": agent.name,
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "tools": [
            {"name": t.name, "description": t.description} for t in agent.tools.all()
        ],
    }


class ChatRequest(BaseModel):
    prompt: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "agent": agent.name, "model": agent.model}


@app.get("/api/agent")
def agent_info() -> dict[str, Any]:
    return _agent_info()


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must be a non-empty string")
    try:
        result = orchestrator.run(agent, prompt)
    except Exception as exc:  # pragma: no cover - defensive: never mask errors
        raise HTTPException(status_code=500, detail=f"agent run failed: {exc}") from exc
    return JSONResponse(_result_to_dict(result))


@app.get("/")
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_file)
