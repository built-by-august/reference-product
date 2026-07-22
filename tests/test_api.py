"""Tests for the lightweight B2C demo web service (CHI-1.5 + CHI-1.3).

Verifies the API surface the demo UI talks to, including the new observability
endpoints: rolling ``/api/metrics`` and the streamed ``/api/chat/stream`` SSE
route. FastAPI's TestClient drives the app in-process.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chi_app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["agent"] and body["model"]
    # Provider selection is surfaced to the dashboard (CHI-9).
    assert "provider" in body


def test_agent_info_exposes_tools() -> None:
    with TestClient(app) as client:
        res = client.get("/api/agent")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "hello"
    assert "provider" in body
    assert any(t["name"] == "echo" for t in body["tools"])


def test_config_exposes_active_provider() -> None:
    with TestClient(app) as client:
        res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert "provider" in body
    assert body["provider"] in {"stub", "openai", "anthropic"}


def test_chat_returns_observed_result() -> None:
    with TestClient(app) as client:
        res = client.post("/api/chat", json={"prompt": "What is Chimeric Intelligence?"})
    assert res.status_code == 200
    body = res.json()
    assert body["content"]
    assert body["trace_id"]
    assert "latency_ms" in body["usage"]
    assert "total_tokens" in body["usage"]
    assert isinstance(body["messages"], list) and body["messages"]


def test_chat_rejects_empty_prompt() -> None:
    with TestClient(app) as client:
        res = client.post("/api/chat", json={"prompt": "   "})
    assert res.status_code == 422


def test_index_served() -> None:
    with TestClient(app) as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_metrics_endpoint_returns_aggregates() -> None:
    with TestClient(app) as client:
        # Generate a run so metrics are non-empty.
        client.post("/api/chat", json={"prompt": "metrics test"})
        res = client.get("/api/metrics")
    assert res.status_code == 200
    body = res.json()
    assert "total_runs" in body
    assert "p95_latency_ms" in body
    assert "total_tokens" in body
    assert "retried_runs" in body
    assert body["total_runs"] >= 1


def test_chat_stream_returns_sse_events() -> None:
    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/chat/stream", json={"prompt": "stream test"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            raw = b"".join(response.iter_bytes()).decode("utf-8")
    # We expect at least one token frame and a final done frame.
    assert "data: " in raw
    assert '"type": "token"' in raw
    assert '"type": "done"' in raw
    # The done frame carries aggregate usage.
    assert "total_tokens" in raw


def test_chat_stream_rejects_empty_prompt() -> None:
    with TestClient(app) as client:
        res = client.post("/api/chat/stream", json={"prompt": "  "})
    assert res.status_code == 422
