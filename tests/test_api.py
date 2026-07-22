"""Tests for the lightweight B2C demo web service (CHI-1.5).

Verifies the API surface the demo UI talks to, without spawning a network
server: FastAPI's TestClient drives the app in-process.
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


def test_agent_info_exposes_tools() -> None:
    with TestClient(app) as client:
        res = client.get("/api/agent")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "hello"
    assert any(t["name"] == "echo" for t in body["tools"])


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
