"""Tool-selection eval.

Given a YAML suite of natural-language tasks each labelled with the expected
MCP tool (and, optionally, expected argument fragments), this eval has the
agent pick one tool per task and grades the result. The headline metric is
accuracy with a Wilson 95% CI; we also persist per-task latency, tokens, cost,
and the picked args so the dashboard and benchmark can dig in further.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mcp_eval.adapters.anthropic_agent import AgentDecision, ToolPicker
from mcp_eval.adapters.mcp_client import MCPClient
from mcp_eval.core.stats import WilsonInterval, wilson_ci
from mcp_eval.core.tracing import get_tracer
from mcp_eval.storage.duckdb_store import DuckDBStore, Result, Run


@dataclass
class TaskCase:
    id: str
    task: str
    expected_tool: str
    expected_args_contain: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutcome:
    case: TaskCase
    decision: AgentDecision | None
    correct: bool
    error: str | None = None


@dataclass
class ToolSelectionReport:
    run_id: str
    model: str
    mcp_server: str
    started_at: datetime
    finished_at: datetime
    outcomes: list[TaskOutcome]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def wilson(self) -> WilsonInterval:
        return wilson_ci(self.correct, self.total)

    def summary(self) -> str:
        ci = self.wilson
        return (
            f"{self.correct}/{self.total} correct  "
            f"({ci.proportion:.1%}, 95% CI [{ci.lower:.2f}, {ci.upper:.2f}])"
        )


def load_yaml_suite(path: str | Path) -> list[TaskCase]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a YAML list of cases")

    cases: list[TaskCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: case {i} is not a mapping")
        if "task" not in item or "expected_tool" not in item:
            raise ValueError(f"{path}: case {i} missing 'task' or 'expected_tool'")
        cid = str(item.get("id") or f"case_{i:03d}")
        expected_args = item.get("expected_args_contain") or {}
        if not isinstance(expected_args, dict):
            raise ValueError(f"{path}: case {cid} expected_args_contain must be a mapping")
        cases.append(
            TaskCase(
                id=cid,
                task=str(item["task"]),
                expected_tool=str(item["expected_tool"]),
                expected_args_contain=expected_args,
            )
        )
    return cases


def _matches_args(actual: Any, expected: Any) -> bool:
    """Recursive partial match.

    Mappings: every expected key must be present in ``actual`` and the values
    must themselves match. Lists: every expected element must equal-match some
    actual element (subset semantics, order-insensitive). Scalars: equality.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and _matches_args(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(any(_matches_args(av, ev) for av in actual) for ev in expected)
    return actual == expected


def grade(decision: AgentDecision, case: TaskCase) -> bool:
    if decision.picked_tool != case.expected_tool:
        return False
    if not case.expected_args_contain:
        return True
    return _matches_args(decision.picked_args or {}, case.expected_args_contain)


def _generate_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


async def run_tool_selection_eval(
    cases: list[TaskCase],
    agent: ToolPicker,
    mcp_client: MCPClient,
    *,
    server_label: str,
    store: DuckDBStore | None = None,
    run_id: str | None = None,
    on_progress: Any | None = None,
) -> ToolSelectionReport:
    """Run the tool-selection eval end to end.

    ``agent`` is anything implementing the :class:`ToolPicker` protocol, so
    tests can pass a deterministic fake. ``mcp_client`` must already be
    connected. If ``store`` is provided, runs and results are persisted.
    """
    tracer = get_tracer()
    started_at = datetime.now(UTC)
    rid = run_id or _generate_run_id()

    if store is not None:
        store.insert_run(
            Run(
                run_id=rid,
                model=agent.model,
                mcp_server=server_label,
                eval_kind="tool_selection",
                started_at=started_at,
            )
        )

    tools = await mcp_client.list_tools()
    outcomes: list[TaskOutcome] = []

    for case in cases:
        with tracer.start_as_current_span("tool_selection.case") as span:
            span.set_attribute("task.id", case.id)
            span.set_attribute("task.expected_tool", case.expected_tool)
            try:
                decision = await asyncio.to_thread(agent.pick_tool, case.task, tools)
            except Exception as exc:
                outcomes.append(
                    TaskOutcome(case=case, decision=None, correct=False, error=str(exc))
                )
                if store is not None:
                    store.insert_result(
                        Result(
                            run_id=rid,
                            task_id=case.id,
                            task_description=case.task,
                            picked_tool=None,
                            expected_tool=case.expected_tool,
                            picked_args=None,
                            expected_args_contain=case.expected_args_contain,
                            correct=False,
                            latency_ms=0.0,
                            error=str(exc),
                        )
                    )
                if on_progress is not None:
                    on_progress(outcomes[-1])
                continue

            correct = grade(decision, case)
            outcome = TaskOutcome(case=case, decision=decision, correct=correct)
            outcomes.append(outcome)
            span.set_attribute("task.correct", correct)
            span.set_attribute("task.picked_tool", decision.picked_tool or "")
            span.set_attribute("task.latency_ms", decision.latency_ms)

            if store is not None:
                store.insert_result(
                    Result(
                        run_id=rid,
                        task_id=case.id,
                        task_description=case.task,
                        picked_tool=decision.picked_tool,
                        expected_tool=case.expected_tool,
                        picked_args=decision.picked_args,
                        expected_args_contain=case.expected_args_contain,
                        correct=correct,
                        latency_ms=decision.latency_ms,
                        tokens_in=decision.tokens_in,
                        tokens_out=decision.tokens_out,
                        cost_usd=decision.cost_usd,
                    )
                )
            if on_progress is not None:
                on_progress(outcome)

    finished_at = datetime.now(UTC)
    if store is not None:
        store.finalize_run(rid, finished_at)

    return ToolSelectionReport(
        run_id=rid,
        model=agent.model,
        mcp_server=server_label,
        started_at=started_at,
        finished_at=finished_at,
        outcomes=outcomes,
    )
