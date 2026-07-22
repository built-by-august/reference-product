"""Lightweight CLI + local dev harness for the Chimeric agent runtime.

Commands:
  chi hello "prompt"   One-shot: run the reference hello agent and print the result.
  chi run              Interactive local harness: type prompts, see traced results,
                       `tools` to list registered tools, `exit`/`quit` to leave.

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

from chi_runtime import Orchestrator
from chi_runtime.agents import build_hello_agent

console = Console()


def _print_result(result) -> None:  # noqa: ANN001 - AgentResult dataclass
    console.print(Panel(result.content, title=f"agent → {result.model}", border_style="cyan"))
    table = Table(title="Observability", show_header=False, box=None)
    table.add_row("trace_id", result.trace_id)
    table.add_row("finish_reason", result.finish_reason.value)
    table.add_row("latency_ms", f"{result.usage.latency_ms:.2f}")
    table.add_row("total_tokens", str(result.usage.total_tokens))
    table.add_row("tool_calls", str(len(result.tool_calls)))
    console.print(table)


def cmd_hello(prompt: str) -> None:
    orchestrator = Orchestrator()
    agent = build_hello_agent()
    result = orchestrator.run(agent, prompt)
    _print_result(result)


def cmd_run() -> NoReturn:
    orchestrator = Orchestrator()
    agent = build_hello_agent()
    console.print(
        Panel(
            f"Chimeric local harness — agent '{agent.name}' (model: {agent.model})\n"
            "Type a prompt and press Enter. Commands: [bold]tools[/bold], [bold]exit[/bold]",
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
        result = orchestrator.run(agent, prompt)
        _print_result(result)


def cmd_serve(host: str, port: int) -> None:
    """Start the lightweight B2C demo web service (CHI-1.5)."""
    import chi_app.main as web

    console.print(
        Panel(
            f"Chimeric B2C demo UI\n"
            f"Open [bold cyan]http://{host}:{port}[/bold cyan] in a browser.\n"
            f"Agent: '{web.agent.name}' · model: {web.agent.model}\n"
            f"Press Ctrl-C to stop.",
            title="chi serve",
            border_style="magenta",
        )
    )
    uvicorn.run(web.app, host=host, port=port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chi", description="Chimeric agent runtime CLI.")
    sub = parser.add_subparsers(dest="command")
    hello = sub.add_parser("hello", help="Run the hello agent once with a prompt.")
    hello.add_argument("prompt", help="The prompt to send to the agent.")
    sub.add_parser("run", help="Start the interactive local dev harness.")
    serve = sub.add_parser("serve", help="Start the B2C demo web UI (FastAPI).")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        cmd_hello(args.prompt)
    elif args.command == "run":
        cmd_run()
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    else:
        build_parser().print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
