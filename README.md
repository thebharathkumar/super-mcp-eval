# mcp-eval

An evaluation harness for [Model Context Protocol](https://modelcontextprotocol.io) servers and the agents that use them.

> **Status:** Phase 1 of 4. Schema-compliance evals are wired up and running in CI. Tool-selection accuracy with Wilson 95% CIs (Phase 2) and McNemar's regression detection (Phase 3) are next.

## What it does today

`mcp-eval` connects to an MCP server over stdio, lists every tool the server advertises, and validates each tool's `inputSchema` against the JSON Schema 2020-12 metaschema. The eval is deterministic: each tool is a binary pass or fail, and the suite exits non-zero if any tool's schema is malformed.

The reference target is Anthropic's public filesystem MCP server (`@modelcontextprotocol/server-filesystem`). Adding another MCP server is a one-line change.

## Quickstart

```bash
git clone https://github.com/thebharathkumar/super-mcp-eval mcp-eval
cd mcp-eval
pip install -e ".[dev]"
pytest
```

The integration test spawns the filesystem MCP via `npx`, so you need Node 18+ on the system path. Without `npx` the integration test is skipped and the unit tests still run.

## Repo layout

```
mcp_eval/
  adapters/mcp_client.py    Stdio MCP client used by every eval
  evals/schema.py           Phase 1: JSON Schema 2020-12 metaschema check per tool
  core/                     (Phase 2) Wilson CI, McNemar's test, eval runner, OTel tracing
  evals/tool_selection.py   (Phase 2) Did Claude pick the right tool for a task?
  evals/regression.py       (Phase 3) Paired model-vs-model regression detection
  storage/duckdb_store.py   (Phase 2) DuckDB persistence for runs and per-task results
  dashboard/app.py          (Phase 2) Streamlit dashboard
tests/                      pytest suite, ``-m integration`` marker for end-to-end runs
benchmarks/                 (Phase 3) generates scorecard.json across reference servers
examples/                   (Phase 2) per-server tool-selection eval YAMLs
```

## Design constraints

- Python 3.11+, pytest-first.
- Anthropic SDK only for v1. Multi-model is deliberately out of scope.
- Local-first: `docker compose up` runs the dashboard, no auth, no multi-tenancy.
- Every number reported in the README and scorecard comes from a real eval run.

## License

MIT.
