"""Streamlit dashboard for browsing mcp-eval runs.

Run with::

    streamlit run mcp_eval/dashboard/app.py

The page shows the list of runs in the configured DuckDB file, lets you pick
one, and renders a scorecard tile (correct/total + Wilson 95% CI), the
per-task table, and a latency histogram for the selected run.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mcp_eval.core.stats import wilson_ci
from mcp_eval.storage.duckdb_store import DuckDBStore, default_db_path

st.set_page_config(page_title="mcp-eval", layout="wide")
st.title("mcp-eval")
st.caption("Evaluation harness for MCP servers and the agents that use them.")

db_path_str = st.sidebar.text_input("DuckDB path", value=str(default_db_path()))
db_path = Path(db_path_str)

if not db_path.exists():
    st.warning(f"No database at `{db_path}`. Run an eval first.")
    st.stop()

store = DuckDBStore(db_path)
runs = store.list_runs()

if not runs:
    st.info("No runs in this database yet. Try `python -m mcp_eval tool-selection --help`.")
    st.stop()

runs_df = pd.DataFrame(runs)
labels = {
    row["run_id"]: f"{row['run_id']}  ({row['model']} on {row['mcp_server']})"
    for _, row in runs_df.iterrows()
}
selected_run_id = st.sidebar.selectbox(
    "Run", options=runs_df["run_id"].tolist(), format_func=lambda r: labels[r]
)

run_row = runs_df[runs_df.run_id == selected_run_id].iloc[0]
st.subheader(f"Run `{selected_run_id}`")
st.write(
    f"**Model:** `{run_row['model']}`  |  "
    f"**MCP server:** `{run_row['mcp_server']}`  |  "
    f"**Eval kind:** `{run_row['eval_kind']}`"
)
st.caption(f"Started {run_row['started_at']} | finished {run_row['finished_at']}")

results = pd.DataFrame(store.get_results(selected_run_id))
total = len(results)
correct = int(results["correct"].sum()) if total else 0
ci = wilson_ci(correct, total)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasks correct", f"{correct} / {total}")
col2.metric("Accuracy", f"{ci.proportion:.1%}")
col3.metric("95% CI", f"[{ci.lower:.2f}, {ci.upper:.2f}]")
total_cost = float(results["cost_usd"].fillna(0).sum()) if total else 0.0
col4.metric("Cost", f"${total_cost:.4f}")

if total > 0:
    st.subheader("Per-task results")
    display = results[
        [
            "task_id",
            "task_description",
            "expected_tool",
            "picked_tool",
            "correct",
            "latency_ms",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "error",
        ]
    ].copy()
    display["latency_ms"] = display["latency_ms"].round(1)
    if "cost_usd" in display:
        display["cost_usd"] = display["cost_usd"].round(5)
    st.dataframe(display, hide_index=True, width="stretch")

    if results["latency_ms"].notna().any():
        st.subheader("Latency distribution")
        chart = (
            alt.Chart(results)
            .mark_bar()
            .encode(
                alt.X("latency_ms:Q", bin=alt.Bin(maxbins=20), title="Latency (ms)"),
                alt.Y("count()", title="Tasks"),
            )
            .properties(height=200)
        )
        st.altair_chart(chart, width="stretch")

store.close()
