"""Command-line entry point for the mcp-eval harness."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Annotated

import typer

from mcp_eval.adapters.anthropic_agent import AnthropicAgent
from mcp_eval.adapters.mcp_client import MCPClient
from mcp_eval.evals.schema import format_report, run_schema_eval
from mcp_eval.evals.tool_selection import (
    load_yaml_suite,
    run_tool_selection_eval,
)
from mcp_eval.storage.duckdb_store import DuckDBStore

app = typer.Typer(no_args_is_help=True, add_completion=False, help="mcp-eval CLI")


def _split_command(mcp_command: str) -> tuple[str, list[str]]:
    parts = shlex.split(mcp_command)
    if not parts:
        raise typer.BadParameter("--mcp-command is empty")
    return parts[0], parts[1:]


@app.command("schema")
def schema(
    mcp_command: Annotated[
        str,
        typer.Option("--mcp-command", help="Stdio command that launches the MCP server."),
    ],
    server_label: Annotated[
        str,
        typer.Option("--server-label", help="Name to attach to the run."),
    ] = "mcp",
) -> None:
    """Validate every tool the MCP server advertises against the JSON Schema 2020-12 metaschema."""
    command, args = _split_command(mcp_command)

    async def _go() -> int:
        async with MCPClient(command=command, args=args) as client:
            report = await run_schema_eval(client, server=server_label)
        typer.echo(format_report(report))
        return 0 if report.all_passed else 1

    raise typer.Exit(asyncio.run(_go()))


@app.command("tool-selection")
def tool_selection(
    suite: Annotated[
        Path,
        typer.Option("--suite", exists=True, dir_okay=False, help="YAML file of TaskCases."),
    ],
    mcp_command: Annotated[
        str,
        typer.Option("--mcp-command", help="Stdio command that launches the MCP server."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Anthropic model id."),
    ] = "claude-sonnet-4-6",
    server_label: Annotated[
        str,
        typer.Option("--server-label"),
    ] = "mcp",
    db: Annotated[
        Path | None,
        typer.Option("--db", help="DuckDB path; overrides MCP_EVAL_DB."),
    ] = None,
) -> None:
    """Pick one MCP tool per task and grade against the YAML expectations."""
    cases = load_yaml_suite(suite)
    command, args = _split_command(mcp_command)

    async def _go() -> int:
        async with MCPClient(command=command, args=args) as client:
            agent = AnthropicAgent(model=model)
            store = DuckDBStore(db) if db else DuckDBStore()
            try:

                def _on_progress(outcome) -> None:
                    marker = "PASS" if outcome.correct else "FAIL"
                    picked = outcome.decision.picked_tool if outcome.decision else "<error>"
                    typer.echo(
                        f"[{marker}] {outcome.case.id}: expected={outcome.case.expected_tool} "
                        f"picked={picked}"
                    )

                report = await run_tool_selection_eval(
                    cases=cases,
                    agent=agent,
                    mcp_client=client,
                    server_label=server_label,
                    store=store,
                    on_progress=_on_progress,
                )
            finally:
                store.close()
        typer.echo("")
        typer.echo(report.summary())
        typer.echo(f"run_id: {report.run_id}")
        return 0 if report.correct == report.total else 1

    raise typer.Exit(asyncio.run(_go()))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
