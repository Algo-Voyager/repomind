# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

All core modules are implemented: `ingest.py`, `tools.py`, `prompts.py`, `logger.py`, `agent.py`, `server.py`, `inngest_setup.py`, `eval/*.py`, and the `frontend/` Next.js app. The UI was migrated from Streamlit to Next.js. Embeddings use Modal (not Ollama). The LLM is Qwen2.5-7B served via rag-learning's Modal deployment over a custom HTTP endpoint (not OpenAI-compatible function calling). When adding features, conform to the module responsibilities below.

## Commands

```bash
# One-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in VLLM_API_KEY, QWEN_GENERATE_URL, EMBED_BASE_URL, GITHUB_TOKEN

# Deploy Modal services from rag-learning (LLM + embeddings)
cd ../rag-learning && modal deploy qwen_modal.py && cd ../repomind

# Terminal 1 — FastAPI + Inngest backend
uvicorn server:app --reload --port 8000

# Terminal 2 — Inngest Dev Server (UI at http://localhost:8288)
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest

# Terminal 3 — Next.js frontend
cd frontend && npm run dev   # http://localhost:3000

# ── Standalone CLI (no server required) ──────────────────────────────────────
python ingest.py <owner/repo> <ast|naive>
python tools.py <owner>_<repo>_<mode>
python agent.py <owner>_<repo>_<mode> "your question here"

# ── Benchmark (requires both AST and naive collections ingested) ─────────────
python eval/compare.py <owner>/<repo>
# Writes results to frontend/public/benchmark_results.json
```

No test runner, linter, or formatter is configured yet. If you add one, update this section.

## Architecture

The system is a from-scratch RAG agent — **no LangChain, no LlamaIndex**. Keep that constraint when adding features; do not introduce those frameworks.

**Data flow:**

1. `ingest.py` pulls a GitHub repo via PyGithub, chunks every file (AST-based for Python, heading-based for Markdown, sliding window for everything else), calls the Modal embedding API (`EMBED_BASE_URL/embeddings`) via the `openai` SDK, and persists vectors to ChromaDB at `./chroma_db`.

2. At query time, `frontend/` sends a `POST /api/query` to `server.py`, which triggers an Inngest `repomind/run_agent` event. The frontend polls `GET /api/result/{event_id}` until the result is ready. The agent in `agent.py` runs a text-based **ReAct loop**: it POSTs to `QWEN_GENERATE_URL` (`httpx`), parses the model's `Action:` / `Action Input:` text output, runs the tool, appends `Observation:` to the prompt, and repeats until `Final Answer:` appears.

3. Conversation history is managed by the frontend in a `contextRef` (per-collection). The backend accepts `history: list[dict]` in the query body, optionally compresses it (at 12 K chars, keeps last 4 pairs verbatim, summarizes the rest with Qwen via `COMPRESS_HISTORY_PROMPT`), and returns `compressed_history` in the result for the frontend to store.

4. Every tool call goes through `tools.py`. `vector_search` embeds the query via the Modal embedding API and queries ChromaDB. Latency (`embed_ms`, `chroma_ms`) is tracked in a `_TOOL_METRICS` ContextVar and reported in `log_step`.

5. After every run, `agent.py` fires a `repomind/agent_completed` event to the Inngest Dev Server (daemon thread, non-blocking). The `agent_completed_fn` in `server.py` runs `compute-metrics` as a step.

## Module responsibilities

- `ingest.py` — repo fetch + chunk + embed + write to Chroma. Exposes `fetch_and_chunk_repo` and `embed_and_store_chunks` for Inngest steps. Runs standalone as CLI. Uses `openai` SDK with `EMBED_BASE_URL` for embeddings — **no Ollama**.
- `agent.py` — text-based ReAct loop via `httpx.post(QWEN_GENERATE_URL)`. Not OpenAI function-calling; parses `Action:` / `Action Input:` text from the model. Exposes `run_agent(user_query, collection_name, history_block="")` and `query_rewrite(user_query, history_context="")`. Fires `repomind/agent_completed` event after every run.
- `tools.py` — tool implementations (`vector_search`, `get_file`, `get_recent_commits`). `TOOL_SCHEMAS` is used for the ReAct prompt description only (not passed to the LLM via API). Tracks embed + Chroma latency via `_TOOL_METRICS` ContextVar.
- `prompts.py` — `REACT_PROMPT_TEMPLATE` (supports `{question}`, `{rewritten}`, `{scratchpad}`, `{history_block}`), `QUERY_REWRITE_PROMPT` (supports `{query}`, `{history_context}`), `COMPRESS_HISTORY_PROMPT` (for the lazy compression step).
- `logger.py` — structured JSONL logging to `agent_logs.jsonl` (gitignored).
- `inngest_setup.py` — shared Inngest client singleton (imported by `server.py` and `agent.py`).
- `server.py` — FastAPI app with Inngest webhook at `/api/inngest`. Three Inngest functions: `repomind/ingest_repo` (2 steps: fetch-and-chunk, embed-and-store), `repomind/run_agent` (compress-history → query-rewrite → llm-generate-N → vector-search-N → …), `repomind/agent_completed` (compute-metrics). REST: `POST /api/ingest`, `POST /api/query`, `GET /api/result/{session_id}`. **All Inngest step handlers are `async def` using `asyncio.to_thread` for blocking I/O** — required to avoid blocking the event loop.
- `frontend/` — Next.js 14 app (App Router). Pages: `/chat`, `/logs`, `/benchmarks`. Key files:
  - `app/layout.tsx` — root layout with `Sidebar` + Plus Jakarta Sans font
  - `app/chat/page.tsx` — framer-motion animated chat; per-collection queue (same collection queues, different collections parallel); `contextRef` for compressed history
  - `components/Sidebar.tsx` — resizable (160–400px drag handle); Lucide icon nav; ingest trigger; indexed repos list
  - `components/ui/animated-ai-chat.tsx` — `AnimatedTextarea`, `TypingDots`, `useAutoResizeTextarea`
  - `lib/api.ts` — `triggerQuery(query, collection, history)`, `pollResult(eventId)`, `fetchCollections()`, `ingestRepo(repo, mode)`
- `eval/compare.py` — AST vs naive benchmark. Uses Modal embeddings (`openai.OpenAI(base_url=EMBED_BASE_URL)`) and Qwen as judge (`QWEN_GENERATE_URL`). Writes to `frontend/public/benchmark_results.json`.
- `eval/metrics.py` — reads `agent_logs.jsonl`, computes per-session + aggregate stats.
- `eval/test_queries.py` — correctness harness over 5 fixed queries with keyword scoring.

## Model and dependencies

- **LLM**: `Qwen/Qwen2.5-7B-Instruct` via rag-learning's `QwenService` on Modal. Endpoint: `QWEN_GENERATE_URL`. Request: `{"prompt": str, "max_new_tokens": int, "temperature": float}`. Response: `{"response": str}`. Called via `httpx.post`. **Not OpenAI-compatible** — no function calling, which is why the agent uses a text-based ReAct loop.
- **Embeddings**: `BAAI/bge-small-en-v1.5` via rag-learning's `embedding_api` on Modal. Endpoint: `EMBED_BASE_URL` (must include `/v1` suffix). OpenAI-compatible `/v1/embeddings`. Called via `openai.OpenAI(base_url=EMBED_BASE_URL)`. Produces 384-dim vectors.
- **Vector store**: ChromaDB persistent client at `./chroma_db` (gitignored). Rebuild with `python ingest.py`. Switching embedding models requires full re-ingest (vector dimensions change).

## Key env vars

| Variable | Used by | Notes |
|----------|---------|-------|
| `VLLM_API_KEY` | agent.py, tools.py, ingest.py | Shared key for both Modal services |
| `QWEN_GENERATE_URL` | agent.py | rag-learning LLM endpoint |
| `EMBED_BASE_URL` | tools.py, ingest.py, eval/compare.py | Must end with `/v1` |
| `EMBED_MODEL` | tools.py, ingest.py, eval/compare.py | `BAAI/bge-small-en-v1.5` |
| `GITHUB_TOKEN` | ingest.py, tools.py | Repo read access |

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `frontend/public/benchmark_results.json` are gitignored. Never commit them.

## Critical implementation constraints

- **Inngest step handlers must be `async def`** with `asyncio.to_thread(blocking_fn, ...)` for any blocking I/O (httpx, chromadb, file I/O). Sync handlers called directly on the asyncio event loop will block all concurrent requests and cause 500 poll failures.
- **No Ollama anywhere.** Embeddings go through `EMBED_BASE_URL` (Modal). The old Ollama calls have been removed.
- **No LangChain / LlamaIndex.** This is intentional; keep all orchestration in `agent.py`'s ReAct loop.
- **Frontend manages history state.** The backend is stateless per-request. `contextRef` in `chat/page.tsx` accumulates compressed history between turns; do not store session state on the server.
