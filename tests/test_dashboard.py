"""Smoke-tests the Streamlit dashboard via Streamlit's programmatic AppTest harness."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from mcp_eval.storage.duckdb_store import DuckDBStore, Result, Run


@pytest.fixture()
def populated_db(tmp_path):
    db_path = tmp_path / "dash.duckdb"
    store = DuckDBStore(db_path)
    started = datetime.now(UTC)
    store.insert_run(
        Run(
            run_id="r1",
            model="claude-sonnet-4-6",
            mcp_server="filesystem",
            eval_kind="tool_selection",
            started_at=started,
        )
    )
    for i in range(10):
        store.insert_result(
            Result(
                run_id="r1",
                task_id=f"t{i:02d}",
                task_description=f"task {i}",
                picked_tool="list_directory" if i < 7 else "read_text_file",
                expected_tool="list_directory",
                picked_args={"path": "/tmp"},
                expected_args_contain={"path": "/tmp"},
                correct=i < 7,
                latency_ms=10.0 + i * 5,
                tokens_in=400 + i,
                tokens_out=20,
                cost_usd=0.0012 + i * 0.0001,
            )
        )
    store.finalize_run("r1", datetime.now(UTC))
    store.close()
    return db_path


def test_dashboard_renders_scorecard_and_table(populated_db, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MCP_EVAL_DB", str(populated_db))
    at = AppTest.from_file(os.path.abspath("mcp_eval/dashboard/app.py")).run()

    assert not at.exception
    assert [t.value for t in at.title] == ["mcp-eval"]

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Tasks correct"] == "7 / 10"
    assert metrics["Accuracy"] == "70.0%"
    assert metrics["95% CI"].startswith("[")
    assert metrics["Cost"].startswith("$")

    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert len(df) == 10
    assert {"task_id", "expected_tool", "picked_tool", "correct", "latency_ms"}.issubset(df.columns)


def test_dashboard_warns_when_db_missing(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MCP_EVAL_DB", str(tmp_path / "missing.duckdb"))
    at = AppTest.from_file(os.path.abspath("mcp_eval/dashboard/app.py")).run()

    assert not at.exception
    assert any("No database" in w.value for w in at.warning)
