# Chimeric Intelligence — Reference Product

Reference product repository for [Chimeric Intelligence](https://github.com/built-by-august/reference-product).
This repo is the canonical starting point for agent-driven products built on our
agent runtime skeleton: an orchestrator + tool-calling scaffold designed to plug in
frontier LLMs through clean, observable interfaces.

## What's in here

- `src/chi_runtime/` — the agent runtime skeleton (models, tool scaffold, agent, orchestrator).
- `src/chi_runtime/agents/hello.py` — a runnable "hello agent" that accepts a prompt and returns a result.
- `src/chi_cli/` — a lightweight CLI + local dev harness (`chi run`, `chi hello`).
- `src/chi_app/` — the lightweight B2C demo web service (FastAPI + single-page UI): `chi serve`.
- `tests/` — fast unit tests for the runtime + CLI + demo API (no network required).
- `.github/workflows/ci.yml` — green CI: lint, type-check, test on every push/PR.

## Quick start

```bash
# 1. Install the toolchain (https://github.com/astral-sh/uv)
brew install uv

# 2. Create the environment + install deps
uv sync

# 3. Run the hello agent from the CLI (stubbed — no model API call)
uv run chi hello "What is Chimeric Intelligence?"

# 4. Run the full local harness (interactive prompt loop)
uv run chi run

# 5. Run the B2C demo web UI (CHI-1.5) — visual-first surface wired to the runtime
uv run chi serve --host 127.0.0.1 --port 8000
#   then open http://127.0.0.1:8000 in a browser

# 6. Run the test suite + linter (same as CI)
uv run pytest
uv run ruff check .
```

## Architecture (skeleton)

```
prompt
  │
  ▼
Agent  ──loads──▶  SystemPrompt + Tool definitions
  │
  ▼
Orchestrator  ──calls──▶  ModelProvider.run(turn)   ◀── pluggable (stub today)
  │                          (observability: trace_id, latency_ms, tokens)
  ▼
Tool-calling loop (scaffold): if the model emits tool calls, execute them and feed
results back. The stub provider returns a canned response, so this path is exercised
by tests without any external API.
  │
  ▼
AgentResult { content, tool_calls, usage, trace_id, finish_reason }
```

### Pluggable model provider

`chi_runtime/model.py` defines a `ModelProvider` protocol. The default
`StubModelProvider` returns deterministic output so the runtime is runnable with zero
config and no cost. To use a real frontier model, implement `ModelProvider` (e.g.
`OpenAIModelProvider`) and inject it into the orchestrator — no other code changes
required. Cost/latency are measured via the `Usage` and `trace` fields on every result.

## Project layout

```
reference-product/
├── README.md
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/
│   ├── chi_runtime/        # agent runtime skeleton
│   │   ├── models.py        # Tool, ToolCall, Message, AgentResult, Usage, etc.
│   │   ├── tool.py          # Tool scaffold + registry
│   │   ├── agent.py         # Agent definition + result type
│   │   ├── model.py         # ModelProvider protocol + StubModelProvider
│   │   ├── orchestrator.py  # run loop (prompt → model → tool calls → result)
│   │   └── agents/
│   │       └── hello.py     # reference "hello agent"
│   └── chi_cli/
│       ├── __init__.py
│       └── main.py          # `chi run` harness + `chi hello` command
└── tests/
    ├── test_runtime.py
    └── test_cli.py
```

## Conventions

- Small, tested increments over big-bang builds.
- Every model call is observable (trace id, latency, token usage).
- Prefer typed protocols + dependency injection so the runtime stays testable offline.

## Roadmap pointer

Parent tracking issue: **CHI-1** (Hire your first engineer and create a hiring plan).
This scaffold satisfies **CHI-3**: runnable repo + green CI + local harness returning a
result from a stubbed agent.
