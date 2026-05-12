"""Streamlit front end for the dev-doc agent.

Three tabs: Chat (drive the agent), Logs (recent activity + aggregate
metrics), Benchmarks (AST vs naive chunking + correctness pass rate).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import chromadb
import httpx
import pandas as pd
import streamlit as st

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
            server_url = os.getenv("REPOMIND_SERVER_URL", "http://localhost:8000")
            try:
                r = httpx.post(
                    f"{server_url}/api/ingest",
                    json={"repo": repo_input, "mode": mode},
                    timeout=10.0,
                )
                r.raise_for_status()
                st.success(
                    f"Ingest triggered for **{repo_input}** ({mode}). "
                    f"Monitor progress → [Inngest Dev UI](http://localhost:8288)"
                )
            except httpx.ConnectError:
                st.error(
                    "Cannot reach repomind server. Start it first:\n\n"
                    "`uvicorn server:app --port 8000`"
                )
            except httpx.TimeoutException:
                st.error(
                    "Server did not respond in time. "
                    "Check that the repomind server is running."
                )
            except httpx.HTTPStatusError as e:
                st.error(f"Ingest failed ({e.response.status_code}): {e.response.text[:200]}")
            except Exception as e:
                st.error(f"Unexpected error triggering ingest: {type(e).__name__}")

    st.divider()

    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = [c.name for c in client.list_collections()]
    except Exception:
        collections = []
        st.warning("Could not load ChromaDB collections.")

    if collections:
        selected = st.selectbox("Select indexed repo", collections)
        try:
            chunk_count = client.get_collection(selected).count()
            st.caption(f"{chunk_count} chunks")
        except Exception:
            st.caption("chunk count unavailable")
    else:
        selected = None
        st.caption("No repos ingested yet")

    st.divider()
    st.caption("💡 Uses Qwen2.5-7B on Modal — monitor runs in the Logs tab")

# ───────────────────────────────── Tabs ─────────────────────────────────────
tab_chat, tab_logs, tab_eval = st.tabs(["💬 Chat", "📜 Logs", "📊 Benchmarks"])

# ─── Tab 1: Chat ────────────────────────────────────────────────────────────
with tab_chat:
    if not selected:
        st.info("Ingest a repo first")
    else:
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}
        if selected not in st.session_state.chat_histories:
            st.session_state.chat_histories[selected] = []

        messages = st.session_state.chat_histories[selected]

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    with st.expander(f"Agent reasoning ({msg['steps']} steps)"):
                        for log in msg.get("logs", []):
                            st.json(log, expanded=False)

        if prompt := st.chat_input("Ask about the codebase..."):
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent working..."):
                    server_url = os.getenv("REPOMIND_SERVER_URL", "http://localhost:8000")
                    result = None
                    msg = None
                    try:
                        # Trigger agent run via Inngest — all steps visible in Dev UI
                        trigger = httpx.post(
                            f"{server_url}/api/query",
                            json={"query": prompt, "collection_name": selected},
                            timeout=10.0,
                        )
                        trigger.raise_for_status()
                        session_id = trigger.json().get("session_id")

                        # Poll for result (2s intervals, up to 4 minutes)
                        for _ in range(120):
                            time.sleep(2)
                            try:
                                poll = httpx.get(
                                    f"{server_url}/api/result/{session_id}",
                                    timeout=5.0,
                                )
                                if poll.status_code == 200:
                                    result = poll.json()
                                    break
                            except Exception:
                                pass

                        if result is None:
                            msg = "The agent did not respond in time. The Modal service may be cold-starting — try again in a moment."
                            st.warning(msg)

                    except httpx.ConnectError:
                        msg = "Cannot reach the repomind server. Check that it is running."
                        st.error(msg)
                    except httpx.TimeoutException:
                        msg = "Server did not respond in time. Check that the repomind server is running."
                        st.error(msg)
                    except httpx.HTTPStatusError as e:
                        msg = f"Server returned an error ({e.response.status_code}). Check the repomind server logs."
                        st.error(msg)
                    except Exception:
                        msg = "Something went wrong while running the agent. Please try again."
                        st.error(msg)

                    if result:
                        session_logs = get_session_logs(result["session_id"])
                        st.markdown(result["answer"])
                        with st.expander(f"Agent reasoning ({result['steps']} steps)"):
                            for log in session_logs:
                                st.json(log, expanded=False)
                        messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "steps": result["steps"],
                            "logs": session_logs,
                        })
                    elif msg:
                        messages.append({"role": "assistant", "content": msg})

# ─── Tab 2: Logs ────────────────────────────────────────────────────────────
with tab_logs:
    st.subheader("Recent agent activity")

    try:
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
    except Exception:
        st.caption("Metrics unavailable")

    st.divider()

    try:
        logs = get_recent_logs(50)
    except Exception:
        logs = []
        st.caption("Could not load logs")

    if logs:
        icons = {
            "tool_call": "🔧",
            "tool_result": "📦",
            "final_answer": "✅",
            "query_rewrite": "✏️",
            "error": "❌",
            "llm_error": "❌",
            "tool_error": "❌",
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
        try:
            data = json.loads(Path("benchmark_results.json").read_text())
            c1, c2, c3 = st.columns(3)
            c1.metric("AST wins", f"{data['ast_wins']}/{data['total_queries']}")
            c2.metric("Avg AST score", data["avg_ast_score"])
            c3.metric("Avg naive score", data["avg_naive_score"])
            df = pd.DataFrame(data["results"])[
                ["query", "ast_avg_score", "naive_avg_score", "winner", "delta"]
            ]
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.warning("benchmark_results.json could not be parsed.")
    else:
        st.info("Run: python eval/compare.py owner/repo")

    st.divider()
    st.subheader("Correctness tests")

    eval_path = Path("eval_results.jsonl")
    if eval_path.exists():
        try:
            results = [
                json.loads(line) for line in eval_path.read_text().splitlines() if line
            ]
            passed_count = sum(1 for r in results if r.get("passed"))
            pass_rate = passed_count / len(results) if results else 0
            st.metric("Pass rate", f"{pass_rate * 100:.0f}%")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        except Exception:
            st.warning("eval_results.jsonl could not be parsed.")
    else:
        st.info("Run: python eval/test_queries.py <collection>")
