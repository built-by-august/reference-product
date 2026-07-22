"""Lightweight CLI + local dev harness for the Chimeric agent runtime.

Commands:
  chi hello "prompt"   One-shot: run the reference hello agent, streaming tokens.
  chi run              Interactive local harness: type prompts, see streamed tokens,
                       `tools` to list registered tools, `exit`/`quit` to leave.
  chi stream "prompt"  One-shot streaming demo: tokens print as they arrive.
  chi serve            Start the B2C demo web UI (FastAPI) with SSE streaming.

Designed so a "visual guy" founder can open a terminal and watch an agent respond,
with latency and token usage surfaced inline (observability by default).
"""

from __future__ import annotations

import argparse
from typing import NoReturn

import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chi_runtime import Orchestrator, build_provider
from chi_runtime.agents import build_hello_agent
from chi_runtime.config import get_config
from chi_runtime.models import StreamEventType

console = Console()


def _build_orchestrator(provider: str | None, model: str | None) -> Orchestrator:
    """Wire the orchestrator to a provider selected at runtime (no code changes elsewhere).

    Precedence: explicit CLI flag  ->  harness ``Config.model``  ->  offline stub.
    API keys resolve via ``Config.model.api_key_env`` then the vendor env var, so a
    real backend can be turned on with env only (no code changes anywhere).
    """
    cfg = get_config()
    mode = (provider or cfg.model.provider or "stub").lower()
    resolved_model = model or (cfg.model.model if cfg.model.model not in ("stub", "") else None)
    return Orchestrator(
        provider=build_provider(
            mode,
            model=resolved_model,
            base_url=cfg.model.api_base,
            api_key_env=cfg.model.api_key_env or "CHI_API_KEY",
            timeout=cfg.model.request_timeout,
        ),
        config=cfg,
    )


def _print_result(result) -> None:  # noqa: ANN001 - AgentResult dataclass
    console.print(Panel(result.content, title=f"agent → {result.model}", border_style="cyan"))
    table = Table(title="Observability", show_header=False, box=None)
    table.add_row("trace_id", result.trace_id)
    table.add_row("finish_reason", result.finish_reason.value)
    table.add_row("latency_ms", f"{result.usage.latency_ms:.2f}")
    table.add_row("total_tokens", str(result.usage.total_tokens))
    table.add_row("tool_calls", str(len(result.tool_calls)))
    console.print(table)


def _stream_prompt(orchestrator: Orchestrator, agent, prompt: str) -> None:
    """Stream a single prompt to the console, tokens as they arrive."""
    console.print(f"[bold cyan]you ›[/bold cyan] {prompt}")
    console.print("[dim]agent ›[/dim] ", end="")
    total_tokens = 0
    latency = 0.0
    finish = "stop"
    trace_id = ""
    for chunk in orchestrator.run_stream(agent, prompt):
        if chunk.event == StreamEventType.TOKEN:
            console.print(chunk.text, end="")
        elif chunk.event == StreamEventType.DONE and chunk.usage:
            total_tokens = chunk.usage.total_tokens
            latency = chunk.usage.latency_ms
            finish = chunk.finish_reason.value if chunk.finish_reason else "stop"
            trace_id = chunk.trace_id or ""
    console.print()  # newline after streamed text
    table = Table(title="Observability (streamed)", show_header=False, box=None)
    table.add_row("trace_id", trace_id)
    table.add_row("finish_reason", finish)
    table.add_row("latency_ms", f"{latency:.2f}")
    table.add_row("total_tokens", str(total_tokens))
    console.print(table)


def cmd_hello(prompt: str, provider: str | None, model: str | None) -> None:
    orchestrator = _build_orchestrator(provider, model)
    agent = build_hello_agent()
    _stream_prompt(orchestrator, agent, prompt)


def cmd_stream(prompt: str, provider: str | None, model: str | None) -> None:
    orchestrator = _build_orchestrator(provider, model)
    agent = build_hello_agent()
    _stream_prompt(orchestrator, agent, prompt)


def cmd_run(provider: str | None, model: str | None) -> NoReturn:
    orchestrator = _build_orchestrator(provider, model)
    agent = build_hello_agent()
    console.print(
        Panel(
            f"Chimeric local harness — agent '{agent.name}' (model: {agent.model})\n"
            "Type a prompt and press Enter. Commands: [bold]tools[/bold], [bold]exit[/bold]\n"
            "Tokens stream as they arrive (CHI-1.3).",
            title="chi run",
            border_style="green",
        )
    )
    while True:
        try:
            prompt = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            raise SystemExit(0) from None
        if prompt.lower() in {"exit", "quit"}:
            console.print("[dim]bye.[/dim]")
            raise SystemExit(0)
        if prompt.lower() == "tools":
            for t in agent.tools.all():
                console.print(f"  • {t.name}: {t.description}")
            continue
        if not prompt:
            continue
        _stream_prompt(orchestrator, agent, prompt)


def cmd_serve(host: str, port: int, provider: str | None = None, model: str | None = None) -> None:
    """Start the lightweight B2C demo web service (CHI-1.5)."""
    import chi_app.main as web

    # Surface the selected provider to the web layer without editing its logic.
    web.orchestrator = _build_orchestrator(provider, model)
    web.agent = build_hello_agent()

    mode = (provider or get_config().model.provider or "stub").lower()
    console.print(
        Panel(
            f"Chimeric B2C demo UI\n"
            f"Open [bold cyan]http://{host}:{port}[/bold cyan] in a browser.\n"
            f"Agent: '{web.agent.name}' · provider: {mode}\n"
            f"Press Ctrl-C to stop.",
            title="chi serve",
            border_style="magenta",
        )
    )
    uvicorn.run(web.app, host=host, port=port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chi", description="Chimeric agent runtime CLI.")

    def add_provider_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--provider",
            default=None,
            choices=["stub", "openai", "anthropic"],
            help="Model backend: stub (offline), openai, or anthropic. "
            "Env override: CHI_PROVIDER.",
        )
        p.add_argument(
            "--model",
            default=None,
            help="Model id override (e.g. gpt-4o-mini, claude-3-5-haiku-latest). "
            "Env override: CHI_MODEL. Defaults to the provider default.",
        )

    sub = parser.add_subparsers(dest="command")
    hello = sub.add_parser("hello", help="Run the hello agent once (streams tokens).")
    hello.add_argument("prompt", help="The prompt to send to the agent.")
    add_provider_args(hello)
    run = sub.add_parser("run", help="Start the interactive local dev harness.")
    add_provider_args(run)
    stream = sub.add_parser("stream", help="Stream a single prompt's tokens to the terminal.")
    stream.add_argument("prompt", help="The prompt to send to the agent.")
    add_provider_args(stream)
    serve = sub.add_parser("serve", help="Start the B2C demo web UI (FastAPI) with SSE.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    add_provider_args(serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        cmd_hello(args.prompt, getattr(args, "provider", None), getattr(args, "model", None))
    elif args.command == "run":
        cmd_run(getattr(args, "provider", None), getattr(args, "model", None))
    elif args.command == "stream":
        cmd_stream(args.prompt, getattr(args, "provider", None), getattr(args, "model", None))
    elif args.command == "serve":
        cmd_serve(
            args.host,
            args.port,
            getattr(args, "provider", None),
            getattr(args, "model", None),
        )
    else:
        build_parser().print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
