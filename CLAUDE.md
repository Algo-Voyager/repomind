# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

All core modules are implemented: `ingest.py`, `tools.py`, `prompts.py`, `logger.py`, `agent.py`, `app.py`, `eval/*.py`. Inngest-based monitoring is active via `server.py` and `inngest_setup.py`. When adding features, conform to the module responsibilities below.

## Commands

Setup assumes a local venv and a running Ollama daemon with `nomic-embed-text` pulled.

```bash
# One-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
cp .env.example .env   # fill in ANTHROPIC_API_KEY, GITHUB_TOKEN

# Terminal 1 — FastAPI + Inngest backend (required for Streamlit ingestion)
uvicorn server:app --reload --port 8000

# Terminal 2 — Inngest Dev Server (UI at http://localhost:8288)
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest

# Terminal 3 — Streamlit UI
streamlit run app.py

# ── Standalone CLI (no server required) ──────────────────────────────────
# Build the local vector index directly (bypasses Inngest)
python ingest.py <owner/repo> <ast|naive>

# Smoke-test the tools against an ingested collection
python tools.py <owner>_<repo>_<mode>

# Run a single agent query from the CLI
python agent.py <owner>_<repo>_<mode> "your question here"
```

No test runner, linter, or formatter is configured yet. If you add one, update this section.

## Architecture

The system is a from-scratch RAG agent — **no LangChain, no LlamaIndex**. Keep that constraint when adding features; do not introduce those frameworks.

Data flow: `ingest.py` pulls a GitHub repo via PyGithub, embeds chunks with Ollama (`nomic-embed-text`), and persists them to ChromaDB at `./chroma_db`. At query time, `app.py` (Streamlit) calls into `agent.py`, which runs an Anthropic tool-use loop against the Claude API. The agent reaches the index through tools defined in `tools.py` — the retrieval path is tool-call mediated, not a direct RAG pipeline, so prompts and tool schemas in `prompts.py` / `tools.py` are the primary surface for behavior changes.

Module responsibilities:
- `ingest.py` — repo fetch + chunk + embed + write to Chroma. Exposes `fetch_and_chunk_repo` and `embed_and_store_chunks` for Inngest steps. Also runs standalone as a CLI.
- `agent.py` — Anthropic SDK conversation loop (`claude-opus-4-7`) with tool dispatch; exposes `run_agent(user_query, collection_name)`. Fires `repomind/agent_completed` to Inngest Dev Server after every run (daemon thread, non-blocking).
- `tools.py` — tool definitions (schemas + implementations). `TOOL_SCHEMAS` is in Claude's format (`{name, description, input_schema}`). Tracks Ollama embed and ChromaDB query latencies via `_TOOL_METRICS` ContextVar.
- `prompts.py` — `SYSTEM_PROMPT` and `QUERY_REWRITE_PROMPT` (the latter has a `{query}` placeholder for `.format()`).
- `logger.py` — structured JSONL logging to `agent_logs.jsonl` (gitignored).
- `inngest_setup.py` — shared Inngest client singleton (imported by `server.py` and `agent.py`).
- `server.py` — FastAPI + Inngest server. Hosts three Inngest functions: `repomind/ingest_repo` (2 steps), `repomind/run_agent` (1 step), `repomind/agent_completed` (post-run metrics). REST: `POST /api/ingest`, `POST /api/query`, `GET /api/result/{session_id}`.
- `app.py` — Streamlit front end. Ingestion triggers `POST /api/ingest` on the server (Inngest background job). Chat runs agent synchronously for immediate UX.
- `deploy/qwen_modal.py` — Modal deployment for Qwen2.5-7B via vLLM. Serves an OpenAI-compatible `/v1/chat/completions` endpoint. `bash cleanup_modal.sh` stops the app and deletes cached volumes.
- `eval/` — offline evaluation harness: `test_queries.py` (benchmark set), `metrics.py` (scoring), `compare.py` (run/config comparison). Writes `eval_results.jsonl` / `benchmark_results.json` (both gitignored).

## Model and dependencies

- LLM: **OpenAI-compatible API** via the `openai` Python SDK pointed at `VLLM_BASE_URL`. Matches rag-learning's pattern. Default model: `Qwen/Qwen2.5-7B-Instruct`. Works with any vLLM-served model that supports OpenAI function-calling (tool use). Set `VLLM_BASE_URL=None` to fall back to the real OpenAI API. `TOOL_SCHEMAS` are in OpenAI format (`{type: "function", function: {name, parameters}}`).
- Embeddings: Ollama `nomic-embed-text`, accessed via the `ollama` Python client — embeddings are local, so the Ollama daemon must be running during ingest and query.
- Vector store: ChromaDB persistent client rooted at `./chroma_db` (gitignored). Treat the directory as disposable; rebuild via `python ingest.py`.

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `benchmark_results.json` are gitignored. Never commit them.

The `ANTHROPIC_API_KEY` is no longer used — repomind now uses `VLLM_API_KEY` + `VLLM_BASE_URL` (OpenAI-compatible).
