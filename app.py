"""Streamlit front end for the dev-doc agent.

Three tabs: Chat (drive the agent), Logs (recent activity + aggregate
metrics), Benchmarks (AST vs naive chunking + correctness pass rate).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st

from agent import run_agent
from eval.metrics import compute_aggregate_metrics
from logger import get_recent_logs, get_session_logs

st.set_page_config(page_title="Dev-Doc Agent", layout="wide")
st.title("Dev-Doc Agent")

# ───────────────────────── Sidebar — repo ingestion ─────────────────────────
with st.sidebar:
    st.header("Repository")

    repo_input = st.text_input(
        "GitHub repo (owner/name)", placeholder="facebook/react"
    )
    mode = st.radio(
        "Chunking mode",
        ["ast", "naive"],
        help="AST = smart code-aware chunking",
    )

    if st.button("Ingest repo", type="primary"):
        if "/" not in repo_input:
            st.error("Format: owner/repo")
        else:
            with st.spinner(f"Ingesting {repo_input} ({mode})..."):
                result = subprocess.run(
                    [sys.executable, "ingest.py", repo_input, mode],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    st.success("Ingested!")
                    st.code(result.stdout[-500:])
                else:
                    st.error(result.stderr[-500:])

    st.divider()

    client = chromadb.PersistentClient(path="./chroma_db")
    collections = [c.name for c in client.list_collections()]

    if collections:
        selected = st.selectbox("Select indexed repo", collections)
        st.caption(f"{client.get_collection(selected).count()} chunks")
    else:
        selected = None
        st.caption("No repos ingested yet")

    st.divider()
    st.caption("💡 Uses Claude API — monitor costs in the Logs tab")

# ───────────────────────────────── Tabs ─────────────────────────────────────
tab_chat, tab_logs, tab_eval = st.tabs(["💬 Chat", "📜 Logs", "📊 Benchmarks"])

# ─── Tab 1: Chat ────────────────────────────────────────────────────────────
with tab_chat:
    if not selected:
        st.info("Ingest a repo first")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    with st.expander(f"Agent reasoning ({msg['steps']} steps)"):
                        for log in msg["logs"]:
                            st.json(log, expanded=False)

        if prompt := st.chat_input("Ask about the codebase..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent working..."):
                    result = run_agent(prompt, selected)
                    session_logs = get_session_logs(result["session_id"])
                    st.markdown(result["answer"])
                    with st.expander(f"Agent reasoning ({result['steps']} steps)"):
                        for log in session_logs:
                            st.json(log, expanded=False)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "steps": result["steps"],
                "logs": session_logs,
            })

# ─── Tab 2: Logs ────────────────────────────────────────────────────────────
with tab_logs:
    st.subheader("Recent agent activity")

    metrics = compute_aggregate_metrics()
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total sessions", metrics["total_sessions"])
        c2.metric("Avg latency", f"{metrics['avg_latency_s']}s")
        c3.metric("Avg steps", metrics["avg_steps"])
        total_tokens = (
            metrics.get("total_input_tokens", 0)
            + metrics.get("total_output_tokens", 0)
        )
        c4.metric("Total tokens", f"{total_tokens:,}")
        c5.metric("Total cost", f"${metrics.get('total_cost_usd', 0):.3f}")

    st.divider()

    logs = get_recent_logs(50)
    if logs:
        icons = {
            "tool_call": "🔧",
            "tool_result": "📦",
            "final_answer": "✅",
            "query_rewrite": "✏️",
            "error": "❌",
            "refusal": "🚫",
            "unexpected_stop": "⚠️",
            "max_steps_reached": "⏱️",
        }
        for log in reversed(logs):
            icon = icons.get(log["event"], "•")
            st.markdown(
                f"{icon} **{log['event']}** · session `{log['session_id']}` "
                f"· step {log['step']} · `{log['timestamp'][11:]}`"
            )
            st.json(log["data"], expanded=False)
    else:
        st.caption("No logs yet")

# ─── Tab 3: Benchmarks ──────────────────────────────────────────────────────
with tab_eval:
    st.subheader("AST vs naive benchmark")

    if Path("benchmark_results.json").exists():
        data = json.loads(Path("benchmark_results.json").read_text())

        c1, c2, c3 = st.columns(3)
        c1.metric("AST wins", f"{data['ast_wins']}/{data['total_queries']}")
        c2.metric("Avg AST score", data["avg_ast_score"])
        c3.metric("Avg naive score", data["avg_naive_score"])

        df = pd.DataFrame(data["results"])[
            ["query", "ast_avg_score", "naive_avg_score", "winner", "delta"]
        ]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run: python eval/compare.py owner/repo")

    st.divider()
    st.subheader("Correctness tests")

    eval_path = Path("eval_results.jsonl")
    if eval_path.exists():
        results = [
            json.loads(line) for line in eval_path.read_text().splitlines() if line
        ]
        passed_count = sum(1 for r in results if r.get("passed"))
        pass_rate = passed_count / len(results) if results else 0
        st.metric("Pass rate", f"{pass_rate * 100:.0f}%")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Run: python eval/test_queries.py <collection>")
