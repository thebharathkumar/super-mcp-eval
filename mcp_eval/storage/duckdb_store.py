"""DuckDB-backed persistence for eval runs and per-task results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    mcp_server TEXT NOT NULL,
    eval_kind TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_description TEXT,
    picked_tool TEXT,
    expected_tool TEXT,
    picked_args TEXT,
    expected_args_contain TEXT,
    correct BOOLEAN NOT NULL,
    latency_ms DOUBLE NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd DOUBLE,
    error TEXT,
    PRIMARY KEY (run_id, task_id)
);
"""


@dataclass
class Run:
    run_id: str
    model: str
    mcp_server: str
    eval_kind: str
    started_at: datetime
    finished_at: datetime | None = None
    notes: str | None = None


@dataclass
class Result:
    run_id: str
    task_id: str
    task_description: str
    picked_tool: str | None
    expected_tool: str | None
    picked_args: dict[str, Any] | None
    expected_args_contain: dict[str, Any] | None
    correct: bool
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None


def default_db_path() -> Path:
    return Path(os.getenv("MCP_EVAL_DB", "./data/eval.duckdb")).expanduser()


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


class DuckDBStore:
    """Thin synchronous wrapper around a single DuckDB file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def insert_run(self, run: Run) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, model, mcp_server, eval_kind, started_at, finished_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.run_id,
                run.model,
                run.mcp_server,
                run.eval_kind,
                run.started_at,
                run.finished_at,
                run.notes,
            ],
        )

    def finalize_run(
        self,
        run_id: str,
        finished_at: datetime,
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, notes = COALESCE(?, notes) WHERE run_id = ?",
            [finished_at, notes, run_id],
        )

    def insert_result(self, result: Result) -> None:
        self._conn.execute(
            """
            INSERT INTO results (
                run_id, task_id, task_description, picked_tool, expected_tool,
                picked_args, expected_args_contain, correct, latency_ms,
                tokens_in, tokens_out, cost_usd, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.run_id,
                result.task_id,
                result.task_description,
                result.picked_tool,
                result.expected_tool,
                _dump(result.picked_args),
                _dump(result.expected_args_contain),
                result.correct,
                result.latency_ms,
                result.tokens_in,
                result.tokens_out,
                result.cost_usd,
                result.error,
            ],
        )

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        cols = [c[0] for c in self._conn.description]
        return [dict(zip(cols, r, strict=False)) for r in rows]

    def get_results(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY task_id",
            [run_id],
        ).fetchall()
        cols = [c[0] for c in self._conn.description]
        return [dict(zip(cols, r, strict=False)) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchall()
        if not rows:
            return None
        cols = [c[0] for c in self._conn.description]
        return dict(zip(cols, rows[0], strict=False))
