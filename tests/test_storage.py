"""Tests for the DuckDB persistence layer."""

from __future__ import annotations

from datetime import UTC, datetime

from mcp_eval.storage.duckdb_store import DuckDBStore, Result, Run


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def test_store_round_trips_run_and_results(tmp_path) -> None:
    db_path = tmp_path / "eval.duckdb"
    store = DuckDBStore(db_path)

    started = _now()
    run = Run(
        run_id="r1",
        model="claude-sonnet-4-6",
        mcp_server="filesystem",
        eval_kind="tool_selection",
        started_at=started,
    )
    store.insert_run(run)

    store.insert_result(
        Result(
            run_id="r1",
            task_id="t1",
            task_description="list /tmp",
            picked_tool="list_directory",
            expected_tool="list_directory",
            picked_args={"path": "/tmp"},
            expected_args_contain={"path": "/tmp"},
            correct=True,
            latency_ms=12.5,
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
        )
    )
    store.insert_result(
        Result(
            run_id="r1",
            task_id="t2",
            task_description="something else",
            picked_tool=None,
            expected_tool="read_text_file",
            picked_args=None,
            expected_args_contain=None,
            correct=False,
            latency_ms=0.0,
            error="boom",
        )
    )
    store.finalize_run("r1", _now(), notes="demo")

    runs = store.list_runs()
    assert [r["run_id"] for r in runs] == ["r1"]
    assert runs[0]["notes"] == "demo"
    assert runs[0]["finished_at"] is not None

    results = store.get_results("r1")
    assert {r["task_id"] for r in results} == {"t1", "t2"}
    by_id = {r["task_id"]: r for r in results}
    assert by_id["t1"]["correct"] is True
    assert by_id["t1"]["picked_args"] == '{"path": "/tmp"}'
    assert by_id["t2"]["error"] == "boom"

    fetched = store.get_run("r1")
    assert fetched is not None
    assert fetched["model"] == "claude-sonnet-4-6"
    assert store.get_run("missing") is None

    store.close()


def test_store_isolates_runs(tmp_path) -> None:
    db_path = tmp_path / "eval.duckdb"
    store = DuckDBStore(db_path)

    for rid in ("a", "b"):
        store.insert_run(
            Run(
                run_id=rid,
                model="m",
                mcp_server="srv",
                eval_kind="tool_selection",
                started_at=_now(),
            )
        )
        store.insert_result(
            Result(
                run_id=rid,
                task_id="t",
                task_description="d",
                picked_tool="x",
                expected_tool="x",
                picked_args={},
                expected_args_contain={},
                correct=True,
                latency_ms=1.0,
            )
        )

    assert len(store.list_runs()) == 2
    assert len(store.get_results("a")) == 1
    assert len(store.get_results("b")) == 1
    assert store.get_results("missing") == []
    store.close()
