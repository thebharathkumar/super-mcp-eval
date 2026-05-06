"""End-to-end smoke for Phase 2 that does not require an Anthropic API key.

Boots the real filesystem MCP server, loads the YAML suite, runs the
tool-selection eval with a deterministic oracle agent that returns the
expected tool for each case, and writes the run to DuckDB. With a real
ANTHROPIC_API_KEY set, swap ``OracleAgent`` for ``AnthropicAgent`` and the
same pipeline produces real numbers.

Usage::

    python scripts/smoke_phase2.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_eval.adapters.anthropic_agent import AgentDecision  # noqa: E402
from mcp_eval.adapters.mcp_client import MCPClient, ToolDef  # noqa: E402
from mcp_eval.evals.tool_selection import (  # noqa: E402
    load_yaml_suite,
    run_tool_selection_eval,
)
from mcp_eval.storage.duckdb_store import DuckDBStore  # noqa: E402


class OracleAgent:
    """Deterministic stub: always returns the expected tool/args for the case."""

    model = "oracle-stub"

    def __init__(self, expected_by_task: dict[str, tuple[str, dict]]) -> None:
        self._expected = expected_by_task

    def pick_tool(self, task: str, tools: list[ToolDef]) -> AgentDecision:
        tool, args = self._expected[task]
        start = time.perf_counter()
        # Sleep a tiny varying amount so the latency distribution in the dashboard is non-degenerate.
        time.sleep(0.005 + (len(task) % 7) * 0.002)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return AgentDecision(
            picked_tool=tool,
            picked_args=dict(args),
            stop_reason="tool_use",
            text=None,
            tokens_in=480 + len(task),
            tokens_out=42,
            cost_usd=(480 + len(task)) / 1_000_000 * 3.0 + 42 / 1_000_000 * 15.0,
            latency_ms=elapsed_ms,
            raw={},
        )


async def main() -> None:
    suite_path = ROOT / "examples" / "filesystem_mcp" / "tool_selection.yaml"
    cases = load_yaml_suite(suite_path)
    print(f"loaded {len(cases)} cases from {suite_path}")

    with tempfile.TemporaryDirectory() as tmproot:
        cmd = shutil.which("mcp-server-filesystem")
        if cmd is None:
            print(
                "mcp-server-filesystem not found; run `npm i -g @modelcontextprotocol/server-filesystem`"
            )
            sys.exit(1)

        oracle = OracleAgent({c.task: (c.expected_tool, c.expected_args_contain) for c in cases})

        db_path = ROOT / "data" / "smoke.duckdb"
        if db_path.exists():
            db_path.unlink()
        store = DuckDBStore(db_path)
        try:
            async with MCPClient(command=cmd, args=[tmproot]) as client:
                report = await run_tool_selection_eval(
                    cases=cases,
                    agent=oracle,
                    mcp_client=client,
                    server_label="filesystem",
                    store=store,
                )
        finally:
            store.close()

    print(f"\n{report.summary()}")
    print(f"run_id: {report.run_id}")
    print(f"db:     {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
