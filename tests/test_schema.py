"""Unit and integration tests for the schema-compliance eval."""

from __future__ import annotations

import shutil

import pytest

from mcp_eval.adapters.mcp_client import MCPClient, ToolDef
from mcp_eval.evals.schema import (
    SchemaReport,
    format_report,
    run_schema_eval,
    validate_tool_schema,
)

FILESYSTEM_PACKAGE = "@modelcontextprotocol/server-filesystem"


def _filesystem_mcp_command(target_dir: str) -> tuple[str, list[str]] | None:
    """Resolve a runnable command for the reference filesystem MCP server.

    Prefers a globally-installed binary so the test works in offline sandboxes
    and warm CI caches. Falls back to ``npx -y`` so the test still works on a
    cold checkout with internet access.
    """
    direct = shutil.which("mcp-server-filesystem")
    if direct:
        return direct, [target_dir]
    npx = shutil.which("npx")
    if npx:
        return npx, ["-y", FILESYSTEM_PACKAGE, target_dir]
    return None


def _tool(name: str, schema: dict) -> ToolDef:
    return ToolDef(name=name, description="", input_schema=schema)


def test_valid_object_schema_passes() -> None:
    tool = _tool(
        "list_directory",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    result = validate_tool_schema(tool)
    assert result.passed
    assert result.message == "ok"


def test_empty_schema_fails() -> None:
    result = validate_tool_schema(_tool("empty", {}))
    assert not result.passed
    assert "empty" in result.message


def test_non_object_top_level_fails() -> None:
    result = validate_tool_schema(_tool("scalar", {"type": "string"}))
    assert not result.passed
    assert "object" in result.message


def test_non_dict_schema_fails() -> None:
    bad = ToolDef(name="weird", description="", input_schema=None)  # type: ignore[arg-type]
    result = validate_tool_schema(bad)
    assert not result.passed
    assert "JSON object" in result.message


def test_invalid_metaschema_fails() -> None:
    # `properties` must itself be an object per the JSON Schema 2020-12 metaschema.
    result = validate_tool_schema(_tool("bad", {"type": "object", "properties": "nope"}))
    assert not result.passed
    assert "JSON Schema" in result.message


def test_schema_report_aggregates() -> None:
    checks = [
        validate_tool_schema(_tool("good", {"type": "object"})),
        validate_tool_schema(_tool("bad", {"type": "string"})),
    ]
    report = SchemaReport(server="fake", checks=checks)
    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert not report.all_passed
    summary = format_report(report)
    assert "1/2 passed" in summary
    assert "[PASS] good" in summary
    assert "[FAIL] bad" in summary


def test_schema_report_serializes() -> None:
    report = SchemaReport(
        server="x",
        checks=[validate_tool_schema(_tool("t", {"type": "object"}))],
    )
    payload = report.as_dict()
    assert payload["server"] == "x"
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["all_passed"] is True
    assert payload["checks"][0]["tool_name"] == "t"


@pytest.mark.integration
async def test_filesystem_mcp_schema(tmp_path) -> None:
    """End-to-end check against Anthropic's reference filesystem MCP server."""
    cmd = _filesystem_mcp_command(str(tmp_path))
    if cmd is None:
        pytest.skip("neither mcp-server-filesystem nor npx is installed")
    command, args = cmd

    async with MCPClient(command=command, args=args) as client:
        report = await run_schema_eval(client, server="filesystem")

    assert report.total > 0, "filesystem MCP returned zero tools"
    failures = [(c.tool_name, c.message) for c in report.checks if not c.passed]
    assert not failures, f"schema failures: {failures}"
    print("\n" + format_report(report))
