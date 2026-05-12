"""FastAPI + Inngest background job server for repomind.

Two Inngest functions:
  repomind/ingest_repo  — fetch-and-chunk → validate-chunks →
                          embed-and-store → log-summary
  repomind/run_agent    — query-rewrite → llm-generate-N →
                          vector-search-N / <tool>-N → ... →
                          check-anomalies → log-summary

Both Streamlit (via POST /api/query + poll) and direct API calls go through
the same Inngest functions, so every run has full per-step visibility in the
Inngest Dev UI.

REST endpoints:
  POST /api/ingest             — trigger a repo ingest job
  POST /api/query              — trigger an agent run, returns event_id
  GET  /api/result/{event_id}  — poll for the agent result

Run with:
    uvicorn server:app --reload --port 8000

Then start the Inngest Dev Server in another terminal:
    npx inngest-cli@latest dev -u http://localhost:8000/api/inngest

Inngest Dev UI is available at http://localhost:8288
"""
from __future__ import annotations

import logging
import uuid

import inngest
import inngest.fast_api
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import chromadb

from eval.metrics import compute_aggregate_metrics
from ingest import embed_and_store_chunks, fetch_and_chunk_repo
from inngest_setup import inngest_client
from logger import get_recent_logs, get_session_logs
from prompts import COMPRESS_HISTORY_PROMPT

logger = logging.getLogger("uvicorn")

# In-memory result cache keyed by event_id (used as session_id in run_agent_fn).
_RESULT_CACHE: dict[str, dict] = {}

# ─── History compression constants ──────────────────────────────────────────
_HISTORY_COMPRESS_THRESHOLD = 12_000   # chars — trigger compression above this
_HISTORY_CHAR_LIMIT = 16_000           # hard cap sent from frontend
_HISTORY_KEEP_RECENT = 4               # message pairs kept verbatim after compress


def _total_history_chars(history: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in history)


def _format_history_block(history: list[dict]) -> str:
    """Format history as 'User: ...\nAssistant: ...' text for LLM consumption."""
    lines = []
    for m in history:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ─── Inngest function 1: ingest repo ────────────────────────────────────────

@inngest_client.create_function(
    fn_id="repomind-ingest-repo",
    trigger=inngest.TriggerEvent(event="repomind/ingest_repo"),
)
async def ingest_repo_fn(ctx: inngest.Context) -> dict:
    """Fetch a GitHub repo, chunk, embed, and store in ChromaDB.

    Steps:
      fetch-and-chunk  — walk repo, apply AST/naive chunking, write temp JSONL
      validate-chunks  — flag empty result before spending time on embeddings
      embed-and-store  — embed every chunk via Modal, upsert ChromaDB
      log-summary      — structured log with totals, errors, and latency
    """
    repo_slug: str = ctx.event.data["repo"]
    mode: str = ctx.event.data.get("mode", "ast")
    event_id: str = ctx.event.id

    # Step 1: fetch and chunk — async so PyGithub HTTP calls don't block the loop
    async def _fetch_and_chunk() -> dict:
        import asyncio as _aio
        return await _aio.to_thread(fetch_and_chunk_repo, repo_slug, mode, event_id)  # positional OK

    chunks_data = await ctx.step.run("fetch-and-chunk", _fetch_and_chunk)

    # Step 2: validate before embedding
    def _validate(d: dict) -> dict:
        issues = []
        if d.get("files_seen", 0) == 0:
            issues.append("no_files_found")
        if d.get("total_chunks", 0) == 0:
            issues.append("no_chunks_produced")
        if issues:
            logger.warning(
                "repomind/ingest_repo validate: repo=%s issues=%s",
                repo_slug, issues,
            )
        return {"valid": len(issues) == 0, "issues": issues}

    validated = await ctx.step.run("validate-chunks", lambda: _validate(chunks_data))

    # Step 3: embed and store — async so Modal embed API calls don't block the loop
    async def _embed_and_store() -> dict:
        import asyncio as _aio
        return await _aio.to_thread(embed_and_store_chunks, chunks_data)

    result = await ctx.step.run("embed-and-store", _embed_and_store)

    # Step 4: log summary
    def _log_summary(r: dict, v: dict) -> dict:
        embed_errors = r.get("embed_errors", 0)
        msg = (
            f"repomind/ingest_repo done: repo={repo_slug} "
            f"collection={r['collection_name']} chunks={r['total_chunks']} "
            f"files={r.get('files_seen', 0)} embed_errors={embed_errors}"
        )
        if v["issues"] or embed_errors > 0:
            logger.warning(msg + f" issues={v['issues']}")
        else:
            logger.info(msg)
        return {"logged": True}

    await ctx.step.run("log-summary", lambda: _log_summary(result, validated))
    return result


# ─── Inngest function 2: agent run ──────────────────────────────────────────

@inngest_client.create_function(
    fn_id="repomind-run-agent",
    trigger=inngest.TriggerEvent(event="repomind/run_agent"),
)
async def run_agent_fn(ctx: inngest.Context) -> dict:
    """ReAct loop driven step-by-step — each LLM call and tool call is a
    separate timed checkpoint in the Inngest Dev UI.

    Steps:
      query-rewrite       — compact semantic-search rewrite of the user query
      llm-generate-N      — Qwen generates Thought + Action (or Final Answer)
      vector-search-N /   — tool execution; embed_ms + chroma_ms captured
        <tool>-N
      check-anomalies     — flag max_steps_reached, no_action, high latency
      log-summary         — structured log with steps, latency, stop reason
    """
    import time
    from agent import _generate, _parse_action, query_rewrite
    from logger import log_step
    from prompts import REACT_PROMPT_TEMPLATE
    from tools import _TOOL_METRICS, run_tool

    query: str = ctx.event.data["query"]
    collection_name: str = ctx.event.data["collection_name"]
    # session_id is generated by /api/query and passed in event data so the
    # caller can poll /api/result/{session_id} without depending on Inngest's
    # internal event ID format.
    session_id: str = ctx.event.data.get("session_id") or ctx.event.id
    history: list[dict] = ctx.event.data.get("history", [])
    run_start = time.time()

    # Step: compress-history — always runs so Inngest replay order is stable.
    # Only makes an LLM call when total history chars exceed the threshold.
    async def _compress_history() -> dict:
        import asyncio as _aio
        total = _total_history_chars(history)
        if total <= _HISTORY_COMPRESS_THRESHOLD or not history:
            return {"history": history, "compressed": False, "total_chars": total}

        # Keep the last KEEP_RECENT pairs verbatim, summarise everything older.
        keep = _HISTORY_KEEP_RECENT * 2
        recent = history[-keep:]
        old = history[:-keep]
        if not old:
            return {"history": history, "compressed": False, "total_chars": total}

        old_text = _format_history_block(old)
        summary = await _aio.to_thread(
            _generate,
            COMPRESS_HISTORY_PROMPT.format(history_text=old_text),
            200,
            0.1,
        )
        compressed = [
            {"role": "assistant", "content": f"[Summary of earlier conversation]: {summary.strip()}"}
        ] + recent
        log_step(session_id, 0, "history_compressed", {
            "original_chars": total,
            "compressed_chars": _total_history_chars(compressed),
            "messages_removed": len(old),
        })
        return {"history": compressed, "compressed": True, "total_chars": _total_history_chars(compressed)}

    compress_result: dict = await ctx.step.run("compress-history", _compress_history)
    effective_history: list[dict] = compress_result["history"]

    # Build the history block injected into the ReAct prompt and the rewrite prompt.
    history_text = _format_history_block(effective_history)
    history_block = (
        f"\nConversation history:\n{history_text}\n"
        if history_text else ""
    )
    # Use only the last 4 messages for the rewrite context (sufficient for reference resolution).
    rewrite_context = _format_history_block(effective_history[-4:]) if effective_history else ""

    # Step: query-rewrite — async so the blocking httpx.post doesn't freeze the
    # event loop; log_step runs inside so it only fires once (not on replays).
    async def _query_rewrite() -> str:
        import asyncio as _aio
        import time as _t
        t = _t.time()
        result = await _aio.to_thread(query_rewrite, query, rewrite_context)
        log_step(session_id, 0, "query_rewrite", {
            "original": query,
            "rewritten": result,
            "latency_s": round(_t.time() - t, 2),
        })
        return result

    rewritten: str = await ctx.step.run("query-rewrite", _query_rewrite)

    scratchpad = ""
    answer = "Could not find a complete answer after max steps."
    stop_reason = "max_steps_reached"
    total_embed_ms = 0
    total_chroma_ms = 0
    step_num = 0

    for step_num in range(1, 7):
        prompt = REACT_PROMPT_TEMPLATE.format(
            question=query,
            rewritten=rewritten,
            scratchpad=scratchpad,
            history_block=history_block,
        )

        # Step: llm-generate-N — async to avoid blocking the event loop
        async def _llm_generate(p: str = prompt) -> str:
            import asyncio as _aio
            return await _aio.to_thread(_generate, p, 1000, 0.2)

        raw: str = await ctx.step.run(f"llm-generate-{step_num}", _llm_generate)

        if "Final Answer:" in raw:
            answer = raw.split("Final Answer:", 1)[-1].strip()
            _REACT_TOKENS = ("Thought:", "Action:", "Action Input:", "Observation:")
            answer = "\n".join(
                line for line in answer.splitlines()
                if not any(line.strip().startswith(tok) for tok in _REACT_TOKENS)
            ).strip()
            stop_reason = "final_answer"
            _ans, _sn = answer, step_num
            await ctx.step.run(
                f"log-final-answer-{step_num}",
                lambda: log_step(session_id, _sn, "final_answer", {
                    "answer": _ans,
                    "total_latency_s": round(time.time() - run_start, 2),
                    "total_steps": _sn,
                }) or {},
            )
            break

        parsed = _parse_action(raw)
        if parsed is None:
            answer = raw
            stop_reason = "no_action"
            _sn = step_num
            await ctx.step.run(
                f"log-unexpected-stop-{step_num}",
                lambda: log_step(session_id, _sn, "unexpected_stop", {}) or {},
            )
            break

        tool_name, args = parsed
        step_label = (
            f"vector-search-{step_num}" if tool_name == "vector_search"
            else f"{tool_name}-{step_num}"
        )

        # Tool step: async so embed + chroma calls don't block the event loop.
        # log_step calls are inside so they only fire once across Inngest replays.
        async def _run_tool(tn: str = tool_name, a: str = args, sn: int = step_num) -> dict:
            import asyncio as _aio
            import time as _t
            log_step(session_id, sn, "tool_call", {"tool": tn, "args": a})
            _TOOL_METRICS.set({})
            t = _t.time()
            res = await _aio.to_thread(run_tool, tn, a, collection_name)
            tool_latency = round(_t.time() - t, 2)
            m = _TOOL_METRICS.get({})
            result_str = str(res)
            log_step(session_id, sn, "tool_result", {
                "tool": tn,
                "result_preview": result_str[:200],
                "result_chars": len(result_str),
                "tool_latency_s": tool_latency,
                "embed_ms": m.get("embed_ms"),
                "chroma_ms": m.get("chroma_ms"),
            })
            return {
                "result": result_str,
                "embed_ms": m.get("embed_ms"),
                "chroma_ms": m.get("chroma_ms"),
            }

        # Step: vector-search-N / <tool>-N
        step_out: dict = await ctx.step.run(step_label, _run_tool)
        total_embed_ms += step_out.get("embed_ms") or 0
        total_chroma_ms += step_out.get("chroma_ms") or 0
        scratchpad += raw + f"\nObservation: {step_out['result']}\n\n"

    total_latency_s = round(time.time() - run_start, 2)

    if stop_reason == "max_steps_reached":
        _sn = step_num
        await ctx.step.run(
            "log-max-steps",
            lambda: log_step(session_id, _sn, "max_steps_reached", {
                "total_latency_s": total_latency_s,
            }) or {},
        )

    # Step: check-anomalies
    def _check_anomalies(sr: str, steps: int, latency: float) -> dict:
        flags = []
        if sr in ("max_steps_reached", "no_action"):
            flags.append(f"incomplete_run:{sr}")
        if latency > 120:
            flags.append(f"high_latency:{latency}s")
        return {"flags": flags, "flagged": len(flags) > 0}

    anomalies = await ctx.step.run(
        "check-anomalies",
        lambda: _check_anomalies(stop_reason, step_num, total_latency_s),
    )

    # Step: log-summary
    def _log_summary(sid: str, steps: int, latency: float, sr: str,
                     emb: int, chro: int, flags: list) -> dict:
        msg = (
            f"repomind/run_agent done: session={sid[:8]} steps={steps} "
            f"latency={latency}s stop={sr} "
            f"embed_ms={emb or None} chroma_ms={chro or None}"
        )
        if flags:
            logger.warning(msg + f" flags={flags}")
        else:
            logger.info(msg)
        return {"logged": True}

    await ctx.step.run(
        "log-summary",
        lambda: _log_summary(
            session_id, step_num, total_latency_s, stop_reason,
            total_embed_ms, total_chroma_ms, anomalies["flags"],
        ),
    )

    result = {
        "session_id": session_id,
        "answer": answer,
        "steps": step_num,
        "stop_reason": stop_reason,
        "total_latency_s": total_latency_s,
        "embed_ms": total_embed_ms or None,
        "chroma_ms": total_chroma_ms or None,
        # Return the (possibly compressed) history so the frontend can use it
        # as the context base for the next message in this conversation.
        "compressed_history": effective_history,
    }
    _RESULT_CACHE[session_id] = result
    return result


# ─── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="repomind server")
inngest.fast_api.serve(app, inngest_client, [ingest_repo_fn, run_agent_fn])


class IngestRequest(BaseModel):
    repo: str
    mode: str = "ast"


class QueryRequest(BaseModel):
    query: str
    collection_name: str
    history: list[dict] = []


@app.post("/api/ingest")
async def trigger_ingest(req: IngestRequest):
    """Trigger a background repo ingestion job."""
    if req.repo.count("/") != 1:
        raise HTTPException(status_code=400, detail="repo must be in 'owner/name' form")
    await inngest_client.send(
        inngest.Event(
            name="repomind/ingest_repo",
            data={"repo": req.repo, "mode": req.mode},
        )
    )
    return {"status": "triggered", "repo": req.repo, "mode": req.mode}


@app.post("/api/query")
async def trigger_query(req: QueryRequest):
    """Trigger an agent run. Poll /api/result/{session_id} for the answer."""
    session_id = str(uuid.uuid4())
    await inngest_client.send(
        inngest.Event(
            name="repomind/run_agent",
            data={
                "query": req.query,
                "collection_name": req.collection_name,
                "session_id": session_id,
                "history": req.history,
            },
        )
    )
    return {"status": "triggered", "session_id": session_id}


@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    """Return cached agent result if ready, else 404."""
    result = _RESULT_CACHE.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="result not ready yet")
    return result


@app.get("/api/collections")
async def list_collections():
    """List all ChromaDB collections with chunk counts."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        result = []
        for col in client.list_collections():
            try:
                count = client.get_collection(col.name).count()
            except Exception:
                count = 0
            result.append({"name": col.name, "chunk_count": count})
        return {"collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def recent_logs(limit: int = 50):
    """Return the most recent agent log entries."""
    try:
        return {"logs": get_recent_logs(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/{session_id}")
async def session_logs(session_id: str):
    """Return all log entries for a specific session."""
    try:
        return {"logs": get_session_logs(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def aggregate_metrics():
    """Return aggregate metrics across all sessions."""
    try:
        metrics = compute_aggregate_metrics()
        return metrics or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
