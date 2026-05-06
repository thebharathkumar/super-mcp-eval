"""Schema-compliance eval.

For every tool an MCP server advertises through ``tools/list``, verify that the
tool's ``inputSchema`` is a well-formed JSON Schema 2020-12 document and that
its top-level type is ``object`` (the MCP spec requires this for tool inputs).
The eval is deterministic and the result is binary per tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from mcp_eval.adapters.mcp_client import MCPClient, ToolDef


@dataclass
class SchemaCheck:
    tool_name: str
    passed: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchemaReport:
    server: str
    checks: list[SchemaCheck]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "all_passed": self.all_passed,
            "checks": [c.as_dict() for c in self.checks],
        }


def validate_tool_schema(tool: ToolDef) -> SchemaCheck:
    """Validate one tool's ``inputSchema`` against the JSON Schema 2020-12 metaschema."""
    schema = tool.input_schema
    if not isinstance(schema, dict):
        return SchemaCheck(tool.name, False, "inputSchema is not a JSON object")
    if not schema:
        return SchemaCheck(tool.name, False, "inputSchema is empty")

    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        return SchemaCheck(tool.name, False, f"invalid JSON Schema: {exc.message}")

    declared_type = schema.get("type")
    if declared_type != "object":
        return SchemaCheck(
            tool.name,
            False,
            f"top-level type must be 'object' for MCP tools, got {declared_type!r}",
        )
    return SchemaCheck(tool.name, True, "ok")


async def run_schema_eval(client: MCPClient, server: str = "unknown") -> SchemaReport:
    """List every tool exposed by ``client`` and validate its schema."""
    tools = await client.list_tools()
    checks = [validate_tool_schema(t) for t in tools]
    return SchemaReport(server=server, checks=checks)


def format_report(report: SchemaReport) -> str:
    """Human-readable single-server summary suitable for CI logs."""
    lines = [f"schema eval :: {report.server} :: {report.passed}/{report.total} passed"]
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{marker}] {check.tool_name}: {check.message}")
    return "\n".join(lines)
