"""Lightweight B2C demo web service for the Chimeric reference product.

This is the thin web layer (CHI-1.5) that sits in front of the agent runtime
skeleton. It imports the already-built ``chi_runtime`` orchestrator and the
reference ``hello`` agent, exposes a tiny JSON API, and serves a single-page
visual-first UI from ``static/``.

CHI-1.3 observability hardening adds here:
  * ``/api/metrics`` — rolling aggregates (p95 latency, total tokens, retry
    rate, failure rate) from the in-process ``MetricsStore``.
  * ``/api/chat/stream`` — Server-Sent Events: the agent's tokens are pushed to
    the browser as they arrive, then a final ``done`` event carries the
    aggregate usage. So the dashboard is now real-time, not just post-hoc.

Design notes
------------
* No new agent logic lives here — the runtime is the source of truth, this
  service only calls ``Orchestrator.run`` / ``Orchestrator.run_stream`` and
  renders the ``AgentResult`` (content + observability) back to the browser.
* The same ``Orchestrator`` is shared across requests. Swapping the stub model
  for a real frontier provider is a one-line change in the runtime and is
  picked up here automatically — cost/latency keep flowing through.
* Model and tool wiring are read from the agent at startup so the UI can show
  what the agent is capable of without hard-coding anything.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from chi_runtime import get_config, get_metrics_store
from chi_runtime.agents import build_hello_agent
from chi_runtime.models import AgentResult, FinishReason, Role, StreamEventType, Usage
from chi_runtime.orchestrator import Orchestrator

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Allow overriding the UI directory (handy for local iteration without a rebuild).
STATIC_DIR = Path(os.environ.get("CHI_UI_DIR", str(STATIC_DIR)))

app = FastAPI(title="Chimeric B2C Demo", version="0.2.0")

# One orchestrator + agent for the process. Cheap to build and reused per request.
# ``cmd_serve`` in the CLI may reassign these to inject a different provider.
orchestrator: Orchestrator = Orchestrator()
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


def _provider_mode() -> str:
    """Which backend the orchestrator is currently wired to (for the dashboard)."""
    try:
        return (get_config().model.provider or "stub").lower()
    except Exception:  # pragma: no cover - defensive fallback
        return "stub"


def _agent_info() -> dict[str, Any]:
    return {
        "name": agent.name,
        "model": agent.model,
        "provider": _provider_mode(),
        "system_prompt": agent.system_prompt,
        "tools": [
            {"name": t.name, "description": t.description} for t in agent.tools.all()
        ],
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    """Expose the resolved runtime config + active provider to the dashboard."""
    cfg = get_config()
    return {
        "provider": _provider_mode(),
        "model": cfg.model.model,
        "temperature": cfg.model.temperature,
        "streaming_enabled": cfg.streaming.enabled,
        "observability": cfg.observability.as_dict(),
    }


def _sse(payload: dict[str, Any]) -> bytes:
    """Format a dict as a Server-Sent Events frame."""
    return f"data: {json.dumps(payload, default=str)}\n\n".encode()


class ChatRequest(BaseModel):
    prompt: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "agent": agent.name, "model": agent.model, "provider": _provider_mode()}


@app.get("/api/agent")
def agent_info() -> dict[str, Any]:
    return _agent_info()


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    """Rolling observability aggregates from the in-process metrics store."""
    snap = get_metrics_store().snapshot()
    return JSONResponse(snap.as_dict())


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


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream the agent's response as Server-Sent Events.

    Events:
      {"type": "token", "text": "..."}  — incremental assistant text
      {"type": "done",  "usage": {...}, "finish_reason": "stop", "trace_id": "..."}
      {"type": "error", "message": "..."}
    The run is executed in a thread so the (synchronous) runtime doesn't block
    the event loop; tokens are forwarded as they are produced.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must be a non-empty string")

    async def event_generator():
        try:
            # Drive the synchronous streaming iterator in a worker thread so we
            # can yield to the event loop between chunks.
            def produce(queue: list) -> None:
                for chunk in orchestrator.run_stream(agent, prompt):
                    queue.append(chunk)

            queue: list = []
            await asyncio.get_running_loop().run_in_executor(None, produce, queue)
            for chunk in queue:
                if chunk.event == StreamEventType.TOKEN and chunk.text:
                    yield _sse({"type": "token", "text": chunk.text})
                elif chunk.event == StreamEventType.DONE:
                    usage = chunk.usage
                    yield _sse({
                        "type": "done",
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": usage.completion_tokens if usage else 0,
                            "total_tokens": usage.total_tokens if usage else 0,
                            "latency_ms": round(usage.latency_ms, 2) if usage else 0.0,
                        },
                        "finish_reason": (chunk.finish_reason.value
                                          if chunk.finish_reason else "stop"),
                    })
        except asyncio.CancelledError:  # client disconnected
            return
        except Exception as exc:  # pragma: no cover - defensive
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_file)
