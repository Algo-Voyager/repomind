# Architecture

End-to-end working of repomind — a grounded code-Q&A agent over a GitHub repository.

## What it does

Given a GitHub repo and a natural-language question, repomind finds the relevant code via a local semantic index, then asks an LLM to answer while *only* citing what it retrieved. Retrieval is mediated by tool calls, not fed into the prompt up front — the model decides what to fetch.

## High-level diagram

```
  ┌──────────────────────────────────────────────────────────────┐
  │               User (Streamlit UI / CLI)                      │
  └──────────┬───────────────────────────────────▲───────────────┘
             │ question / ingest trigger          │ answer
             ▼                                   │
  ┌──────────────────────────┐                   │
  │        app.py            │                   │
  │   (Streamlit)            │                   │
  │  • chat → run_agent()    │                   │
  │  • sidebar → POST        │                   │
  │    /api/ingest           │                   │
  └──────┬────────────────┬──┘                   │
         │ sync chat      │ ingest trigger        │
         │                ▼                       │
         │   ┌─────────────────────────────┐      │
         │   │  server.py (FastAPI)        │      │
         │   │  + Inngest webhook          │      │
         │   │                             │      │
         │   │  repomind/ingest_repo ──────┼──────┼─► fetch-and-chunk
         │   │    step 1: fetch-and-chunk  │      │     (GitHub API)
         │   │    step 2: embed-and-store  │      │   embed-and-store
         │   │                             │      │     (Ollama→Chroma)
         │   │  repomind/agent_completed ◄─┼──────┼─── monitoring event
         │   │    step 1: compute-metrics  │      │
         │   └─────────────────────────────┘      │
         │              ▲                          │
         │              │ fire-and-forget event    │
         ▼              │                          │
  ┌────────────────────────────────────────────┐  │
  │                agent.py                    │  │
  │  • query_rewrite (vLLM)                    │  │
  │  • ReAct loop:                             │  │
  │    chat.completions.create(tools=...)      │  │
  │    finish_reason == tool_calls?            │  │
  │       run_tool → ChromaDB / GitHub API     │  │
  │    finish_reason == stop → return answer   │──┘
  └────────────────────────────────────────────┘

  Side channels:
    logger.py → agent_logs.jsonl → eval/metrics.py
                                 → Streamlit Logs tab
    tools.py  → _TOOL_METRICS (ContextVar) → agent_logs.jsonl tool_result entries
    Inngest Dev UI (localhost:8288) ← all job/event traces
```

## Components

| Module | Role |
|---|---|
| `inngest_setup.py` | Shared Inngest client singleton (`app_id="repomind"`, dev mode). Imported by `server.py` and `agent.py`; never imported by `app.py`. |
| `server.py` | FastAPI app with Inngest webhook at `/api/inngest`. Hosts three Inngest functions (`ingest_repo`, `run_agent`, `agent_completed`) and three REST helpers (`POST /api/ingest`, `POST /api/query`, `GET /api/result/{id}`). |
| `ingest.py` | Fetch a repo via PyGithub, chunk files (AST for `.py`, H2 sections for `.md`, naive fixed-size otherwise), embed each chunk with Ollama `nomic-embed-text`, upsert into ChromaDB at `./chroma_db`. Exposes `fetch_and_chunk_repo` and `embed_and_store_chunks` for Inngest steps; `main()` calls both for CLI use. |
| `agent.py` | The orchestrator. Uses the `openai` SDK with `VLLM_BASE_URL` — identical to rag-learning's client pattern. Calls `query_rewrite` once, then loops `chat.completions.create(tools=TOOL_SCHEMAS)` → inspect `finish_reason` → dispatch `tool_calls` → feed `tool` results back → repeat. Fires a `repomind/agent_completed` Inngest event after every run (daemon thread, non-blocking). |
| `tools.py` | Three tools: `vector_search` (Ollama embed → Chroma query), `get_file` (raw file via PyGithub), `get_recent_commits`. `TOOL_SCHEMAS` is in OpenAI function-calling format (`{type: "function", function: {name, parameters}}`). Tracks Ollama embed latency and ChromaDB query latency via `_TOOL_METRICS` ContextVar — read by `agent.py` after each `run_tool` call. |
| `prompts.py` | `SYSTEM_PROMPT` (agent rules + answer format) and `QUERY_REWRITE_PROMPT` (compresses the user question into a semantic-search query). |
| `logger.py` | Appends one JSON line per event to `agent_logs.jsonl`. `get_recent_logs` / `get_session_logs` are readers for the UI. |
| `app.py` | Streamlit UI. Three tabs: **Chat**, **Logs** (aggregate metrics + recent events), **Benchmarks** (AST vs naive + correctness pass rate). Sidebar triggers ingestion via `POST /api/ingest` on the FastAPI server (Inngest background job). Chat runs `run_agent()` synchronously for immediate UX. |
| `eval/metrics.py` | Reads `agent_logs.jsonl` and computes per-session + aggregate stats: latency, tool/LLM counts, input/output tokens, estimated USD cost. Pricing defaults to $0 for local models; overridable via `LLM_INPUT_PRICE_PER_M` / `LLM_OUTPUT_PRICE_PER_M` env vars. |
| `eval/test_queries.py` | Correctness harness — runs 5 fixed queries through the agent and scores each by must-contain / must-not-contain keywords. Writes `eval_results.jsonl`. |
| `eval/compare.py` | AST vs naive benchmark — retrieves top-3 chunks from both collections for 8 queries, uses the LLM as judge to rate each chunk 1–5, writes `benchmark_results.json`. |
| `deploy/qwen_modal.py` | Reference Modal deployment for vLLM. Update the model name and GPU config to match your target. |

## Workflow 1 — Ingestion

### Via Streamlit sidebar (recommended — Inngest-monitored)

1. User enters `owner/repo` and clicks "Ingest repo".
2. `app.py` POSTs to `http://localhost:8000/api/ingest` → `server.py` sends `repomind/ingest_repo` event to Inngest.
3. Inngest runs the function in the background with two steps visible in the Dev UI at localhost:8288:

**Step 1 — `fetch-and-chunk`** (`ingest.fetch_and_chunk_repo`):
   - Load `GITHUB_TOKEN`, resolve the repo via PyGithub.
   - Walk the tree, skipping `node_modules` / `.git` / `dist` / `build` and anything over 500 KB.
   - For each allowed file (`.py`, `.md`, `.ts`, `.tsx`, `.js`, `.jsx`, `.txt`):
     - `.py` + `ast` mode → parse with `ast.walk`, emit one chunk per function / class.
     - `.md` → split at `## ` H2 headings, one chunk per section.
     - Anything else → naive fixed-size window (2000 chars, 50 overlap).
   - Write all chunks to a temp JSONL file at `./chroma_db/.chunks_<collection>_<event_id>.jsonl`. The `event_id` makes the filename idempotent across Inngest retries.
   - Returns `{collection_name, temp_path, files_seen, total_chunks}` (small dict — safe as Inngest step return value).

**Step 2 — `embed-and-store`** (`ingest.embed_and_store_chunks`):
   - Read the temp JSONL chunk-by-chunk.
   - Embed each chunk via Ollama `nomic-embed-text` (local, no network call).
   - Upsert into Chroma collection `<owner>_<repo>_<mode>` with metadata: `type`, `name`, `file_path`, `line_start`, `line_end`, `language`, `heading`.
   - Delete the temp file in a `finally` block (so Inngest retries of step 2 can still find the file if step 2 itself failed mid-way).

**Result:** a persistent Chroma collection at `./chroma_db`. Streamlit sidebar returns immediately ("triggered"); watch progress in Inngest Dev UI.

### Via CLI (no server required)

```bash
python ingest.py <owner>/<repo> <ast|naive>
```

Calls `fetch_and_chunk_repo` and `embed_and_store_chunks` sequentially. Identical logic, no Inngest visibility.

## Workflow 2 — Query (`python agent.py <collection> "question"` or UI Chat tab)

1. **Query rewrite.** `agent.query_rewrite` sends the user question to the vLLM endpoint with `QUERY_REWRITE_PROMPT` and gets back a compact semantic query (e.g. *"how do I log in?"* → *"authentication login flow implementation"*).

2. **Seed the conversation.** Messages start with the system prompt followed by a user turn containing both the original and rewritten questions.

3. **ReAct loop** (up to `max_steps=6`):
   1. `client.chat.completions.create(model=VLLM_MODEL, tools=TOOL_SCHEMAS, tool_choice="auto", messages=...)`.
   2. If `finish_reason == "stop"` → extract `message.content`, log it with token counts, fire monitoring event, return.
   3. If `finish_reason == "tool_calls"` → for each `tool_call` in `message.tool_calls`:
      - Parse `tc.function.name` and `json.loads(tc.function.arguments)`.
      - Reset `_TOOL_METRICS` ContextVar, call `run_tool(name, args, collection_name)`.
      - Read back `embed_ms` / `chroma_ms` from `_TOOL_METRICS`, merge into the `tool_result` log entry.
      - Append `{role: "tool", tool_call_id: tc.id, content: result_str}` to messages.
   4. Append the full assistant message (with `tool_calls` list) before the tool results.
   5. Loop.
   4. If any other `finish_reason` (length, content_filter, etc.) → surface as `unexpected_stop`, return.

4. **Monitoring.** After every return path, `_fire_monitoring_event` starts a daemon thread that POSTs `repomind/agent_completed` to the Inngest Dev Server at `INNGEST_DEV_EVENT_URL`. The Inngest function runs a `compute-metrics` step, making the run visible in the Dev UI with cost/latency/token breakdown.

5. **Logging.** Every LLM call, tool call (with args + LLM latency), tool result (with embed_ms, chroma_ms, tool latency), final answer (with total latency + token usage), or error event is appended to `agent_logs.jsonl` via `logger.log_step`.

6. **Return.** Payload: `{session_id, answer, steps, messages, usage: {input_tokens, output_tokens}}`. The UI renders the answer plus an expandable per-step trace from `get_session_logs(session_id)`.

## Workflow 3 — Evaluation

- **Correctness** (`eval/test_queries.py`) — drives the agent over 5 predefined questions and checks each answer for must-contain / must-not-contain keyword lists. Pass rate is printed; per-query rows go to `eval_results.jsonl`.
- **AST vs naive** (`eval/compare.py`) — retrieves top-3 chunks from `<owner>_<repo>_ast` and `<owner>_<repo>_naive` for each of 8 benchmark queries and uses the LLM as judge to score each chunk 1–5. Outputs `benchmark_results.json` with win counts, per-query deltas, and averages. Requires both collections ingested.
- **Aggregate metrics** (`eval/metrics.py`) — reads `agent_logs.jsonl`, computes per-session totals (latency, steps, tool counts, tokens) and an overall summary (avg / median / p95 latency, total tokens, total estimated cost).

## Data stores

| File / directory | Written by | Read by | Gitignored |
|---|---|---|---|
| `./chroma_db/` | `ingest.embed_and_store_chunks` | `tools.vector_search`, `eval/compare.py` | ✅ |
| `./chroma_db/.chunks_*.jsonl` | `ingest.fetch_and_chunk_repo` | `ingest.embed_and_store_chunks` | ✅ (temp, auto-deleted) |
| `agent_logs.jsonl` | `logger.log_step` | `eval/metrics.py`, `app.py` Logs tab | ✅ |
| `eval_results.jsonl` | `eval/test_queries.py` | `app.py` Benchmarks tab | ✅ |
| `benchmark_results.json` | `eval/compare.py` | `app.py` Benchmarks tab | ✅ |
| `.env` | user | all modules needing secrets | ✅ |

## Design choices worth knowing

- **OpenAI-compatible LLM client.** `agent.py` uses the `openai` SDK with `base_url=VLLM_BASE_URL` — the same pattern as rag-learning's `data_loader.py`. Any vLLM-served model that supports OpenAI function-calling (tool use) works without code changes. Set `VLLM_BASE_URL=None` to fall back to the real OpenAI API.

- **Tool-call-mediated retrieval, not a fixed RAG pipeline.** The model decides when to search, when to read a whole file, and when to stop. `prompts.py` and the tool descriptions are the primary behavior-tuning surface.

- **Embeddings are always local.** Ollama runs `nomic-embed-text` on the host — no embedding calls hit the vLLM server or any external vendor. Only the generative LLM calls go over the network.

- **Inngest for ingest, monitoring event for agent.** Repo ingestion runs as a durable 2-step Inngest function (retries, step traces, error UI). Agent queries run synchronously in Streamlit for immediate UX; they fire a fire-and-forget `repomind/agent_completed` event after completion so the same Dev UI shows the run's cost, latency, and token breakdown.

- **`_TOOL_METRICS` ContextVar for sub-tool latency.** Ollama embed latency and ChromaDB query latency are tracked inside `tools.py` using a `contextvars.ContextVar` rather than a module-level dict — this is thread-safe and async-safe, so concurrent Streamlit sessions and the FastAPI server don't race.

- **Collections are versioned by chunking mode.** `<owner>_<repo>_ast` and `<owner>_<repo>_naive` live side by side; this is what makes the AST vs naive benchmark possible without re-ingesting.

- **Idempotent ingest retries.** The temp chunk file is named `<collection>_<inngest_event_id>.jsonl`. If Inngest retries step 1 it overwrites the same file; if it retries step 2 the file is still there (only deleted in `finally` after a successful embed pass).
