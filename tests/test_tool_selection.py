"""Tests for the tool-selection eval: YAML loading, grading, and end-to-end run with a fake agent."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from mcp_eval.adapters.anthropic_agent import AgentDecision
from mcp_eval.adapters.mcp_client import ToolDef
from mcp_eval.evals.tool_selection import (
    TaskCase,
    ToolSelectionReport,
    _matches_args,
    grade,
    load_yaml_suite,
    run_tool_selection_eval,
)
from mcp_eval.storage.duckdb_store import DuckDBStore


def _decision(
    tool: str | None, args: dict | None = None, latency_ms: float = 12.5
) -> AgentDecision:
    return AgentDecision(
        picked_tool=tool,
        picked_args=dict(args) if args is not None else None,
        stop_reason="tool_use",
        text=None,
        tokens_in=42,
        tokens_out=7,
        cost_usd=0.000123,
        latency_ms=latency_ms,
        raw={},
    )


def test_load_yaml_suite_assigns_default_ids(tmp_path) -> None:
    suite = tmp_path / "cases.yaml"
    suite.write_text(
        '- id: t1\n  task: "do thing"\n  expected_tool: "do"\n'
        "  expected_args_contain:\n    x: 1\n"
        '- task: "second"\n  expected_tool: "noop"\n'
    )
    cases = load_yaml_suite(suite)
    assert len(cases) == 2
    assert cases[0].id == "t1"
    assert cases[0].expected_args_contain == {"x": 1}
    assert cases[1].id == "case_001"


def test_load_yaml_suite_rejects_missing_keys(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- task: missing expected_tool\n")
    with pytest.raises(ValueError):
        load_yaml_suite(bad)


def test_load_yaml_suite_rejects_non_list(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("just_a_string")
    with pytest.raises(ValueError):
        load_yaml_suite(bad)


def test_filesystem_yaml_suite_loads_at_least_twenty() -> None:
    cases = load_yaml_suite("examples/filesystem_mcp/tool_selection.yaml")
    assert len(cases) >= 20
    ids = [c.id for c in cases]
    assert len(set(ids)) == len(ids), "duplicate task IDs in suite"


def test_grade_correct_when_tool_and_args_match() -> None:
    case = TaskCase(id="x", task="t", expected_tool="list", expected_args_contain={"path": "/tmp"})
    assert grade(_decision("list", {"path": "/tmp"}), case)


def test_grade_wrong_tool_fails() -> None:
    case = TaskCase(id="x", task="t", expected_tool="list", expected_args_contain={})
    assert not grade(_decision("read", {}), case)


def test_grade_wrong_arg_value_fails() -> None:
    case = TaskCase(id="x", task="t", expected_tool="list", expected_args_contain={"path": "/tmp"})
    assert not grade(_decision("list", {"path": "/var"}), case)


def test_grade_extra_args_are_allowed() -> None:
    case = TaskCase(id="x", task="t", expected_tool="list", expected_args_contain={"path": "/tmp"})
    assert grade(_decision("list", {"path": "/tmp", "verbose": True}), case)


def test_grade_no_args_required_passes_with_any_args() -> None:
    case = TaskCase(id="x", task="t", expected_tool="list_allowed_directories")
    assert grade(_decision("list_allowed_directories", {}), case)


def test_matches_args_dict_is_subset() -> None:
    assert _matches_args({"a": 1, "b": 2}, {"a": 1})
    assert not _matches_args({"a": 1}, {"a": 1, "b": 2})


def test_matches_args_list_subset_order_insensitive() -> None:
    assert _matches_args(["a", "b", "c"], ["c", "a"])
    assert not _matches_args(["a"], ["a", "b"])


def test_matches_args_recurses_into_nested_dict() -> None:
    assert _matches_args({"x": {"y": 1, "z": 2}}, {"x": {"y": 1}})
    assert not _matches_args({"x": {"y": 1}}, {"x": {"y": 1, "z": 2}})


class _FakeMCPClient:
    """Mimics MCPClient.list_tools but never spawns a process."""

    def __init__(self, tools: list[ToolDef]) -> None:
        self._tools = tools

    async def list_tools(self) -> list[ToolDef]:
        return self._tools


class _ScriptedAgent:
    model = "stub-model"

    def __init__(self, decisions_by_task: dict[str, AgentDecision]) -> None:
        self._decisions = decisions_by_task

    def pick_tool(self, task: str, tools: list[ToolDef]) -> AgentDecision:
        return self._decisions[task]


async def test_end_to_end_run_persists_to_duckdb(tmp_path) -> None:
    cases = [
        TaskCase(
            id="ok",
            task="list /tmp",
            expected_tool="list_directory",
            expected_args_contain={"path": "/tmp"},
        ),
        TaskCase(
            id="bad",
            task="open notes",
            expected_tool="read_text_file",
            expected_args_contain={"path": "/tmp/n.txt"},
        ),
    ]
    decisions = {
        "list /tmp": _decision("list_directory", {"path": "/tmp"}),
        "open notes": _decision("write_file", {"path": "/tmp/n.txt"}),
    }
    agent = _ScriptedAgent(decisions)
    client = _FakeMCPClient([ToolDef("list_directory", "", {"type": "object"})])
    store = DuckDBStore(tmp_path / "eval.duckdb")
    try:
        report = await run_tool_selection_eval(
            cases=cases,
            agent=agent,  # type: ignore[arg-type]
            mcp_client=client,  # type: ignore[arg-type]
            server_label="fake",
            store=store,
        )
    finally:
        # Re-open in a moment to query, so use a fresh handle.
        pass

    assert report.total == 2
    assert report.correct == 1
    assert math.isclose(report.accuracy, 0.5)
    assert 0.0 < report.wilson.lower < 0.5 < report.wilson.upper < 1.0
    assert report.summary().startswith("1/2 correct")

    persisted_runs = store.list_runs()
    assert len(persisted_runs) == 1
    assert persisted_runs[0]["model"] == "stub-model"
    persisted_results = store.get_results(report.run_id)
    assert {r["task_id"] for r in persisted_results} == {"ok", "bad"}
    correct_map = {r["task_id"]: r["correct"] for r in persisted_results}
    assert correct_map == {"ok": True, "bad": False}
    store.close()


async def test_run_records_agent_exception_as_failed_outcome(tmp_path) -> None:
    class _ExplodingAgent:
        model = "boom"

        def pick_tool(self, task: str, tools: list[ToolDef]) -> AgentDecision:
            raise RuntimeError("simulated 429")

    case = TaskCase(id="t", task="x", expected_tool="list_directory")
    client = _FakeMCPClient([ToolDef("list_directory", "", {"type": "object"})])
    store = DuckDBStore(tmp_path / "eval.duckdb")
    report = await run_tool_selection_eval(
        cases=[case],
        agent=_ExplodingAgent(),  # type: ignore[arg-type]
        mcp_client=client,  # type: ignore[arg-type]
        server_label="fake",
        store=store,
    )
    assert report.total == 1
    assert report.correct == 0
    assert report.outcomes[0].error == "simulated 429"
    persisted = store.get_results(report.run_id)
    assert persisted[0]["error"] == "simulated 429"
    store.close()


def test_tool_selection_report_summary_format() -> None:
    started = finished = datetime.now(UTC)
    report = ToolSelectionReport(
        run_id="r",
        model="m",
        mcp_server="s",
        started_at=started,
        finished_at=finished,
        outcomes=[],
    )
    assert report.summary().startswith("0/0")
