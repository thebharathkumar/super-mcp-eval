# mcp-eval

An evaluation harness for [Model Context Protocol](https://modelcontextprotocol.io) servers and the agents that use them.

> **Status:** Phase 2 of 4 complete. Schema compliance, tool-selection accuracy with Wilson 95% CIs, DuckDB persistence, and a Streamlit dashboard all run end to end. Cross-model regression detection with McNemar's test (Phase 3) and the published benchmark scorecard are next.

## What it does today

1. **Schema compliance** (deterministic). Connects to an MCP server over stdio, lists every tool, and validates each `inputSchema` against the JSON Schema 2020-12 metaschema. Binary pass/fail per tool.
2. **Tool selection** (semantic). Reads a YAML suite of natural-language tasks, asks Claude to pick exactly one MCP tool per task, and grades the picked tool name plus an optional partial match on the picked arguments. Reports accuracy with a Wilson 95% confidence interval and persists per-task latency, tokens, and cost to DuckDB.
3. **Dashboard.** A Streamlit page browses runs, renders the scorecard tile (correct/total + Wilson 95% CI + cost), the per-task table, and a latency histogram.

The reference target is Anthropic's public filesystem MCP server (`@modelcontextprotocol/server-filesystem`); the bundled YAML suite has 24 tasks covering all 14 of its tools.

## Quickstart

```bash
git clone https://github.com/thebharathkumar/super-mcp-eval mcp-eval
cd mcp-eval
pip install -e ".[dev]"
pytest                                   # unit + integration

# Run the schema eval against the filesystem MCP.
python -m mcp_eval schema \
    --mcp-command "npx -y @modelcontextprotocol/server-filesystem /tmp" \
    --server-label filesystem

# Run the tool-selection eval against the filesystem MCP using Claude.
export ANTHROPIC_API_KEY=sk-ant-...
python -m mcp_eval tool-selection \
    --suite examples/filesystem_mcp/tool_selection.yaml \
    --mcp-command "npx -y @modelcontextprotocol/server-filesystem /tmp" \
    --model claude-sonnet-4-6 \
    --server-label filesystem

# Browse runs in the dashboard.
streamlit run mcp_eval/dashboard/app.py
```

The schema integration test runs against the live filesystem MCP via `npx`. Without Node, it is skipped and unit tests still run.

## Repo layout

```
mcp_eval/
  adapters/mcp_client.py        Stdio MCP client used by every eval
  adapters/anthropic_agent.py   Single-shot Claude tool picker (Anthropic Messages API + tool_choice=any)
  evals/schema.py               JSON Schema 2020-12 metaschema check per tool
  evals/tool_selection.py       YAML cases -> agent.pick_tool -> grade -> DuckDB
  evals/regression.py           (Phase 3) Paired model-vs-model regression detection
  core/stats.py                 Wilson CI + McNemar's test (binomial-exact + chi-squared)
  core/tracing.py               OpenTelemetry tracer (no-op by default; MCP_EVAL_TRACE_CONSOLE=1 to dump)
  storage/duckdb_store.py       runs and results tables, per-run query helpers
  dashboard/app.py              Streamlit scorecard, per-task table, latency histogram
  __main__.py                   typer CLI: `python -m mcp_eval schema|tool-selection`
tests/                          pytest suite (unit + integration markers)
examples/filesystem_mcp/        24 tool-selection cases for the reference server
benchmarks/                     (Phase 3) reference-server scorecard.json
scripts/smoke_phase2.py         End-to-end smoke that runs the YAML suite without an Anthropic key
```

## Design constraints

- Python 3.11+, pytest-first. ruff + black, no exceptions.
- Anthropic SDK only for v1. Multi-model is deliberately out of scope.
- Local-first: `docker compose up` runs the dashboard, no auth, no multi-tenancy.
- Every number reported in the README, dashboard, and scorecard comes from a real eval run.

## License

MIT.
