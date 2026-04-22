# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`ingest.py`, `tools.py`, `prompts.py`, `logger.py`, and `agent.py` are implemented. `app.py` and `eval/*.py` are still empty scaffolds. When adding code, conform to the module responsibilities described below rather than inventing a new layout.

## Commands

Setup assumes a local venv and a running Ollama daemon with `nomic-embed-text` pulled.

```bash
# One-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
cp .env.example .env   # fill in ANTHROPIC_API_KEY, GITHUB_TOKEN

# Build the local vector index (writes to ./chroma_db)
python ingest.py <owner/repo> <ast|naive>

# Smoke-test the tools against an ingested collection
python tools.py <owner>_<repo>_<mode>

# Run a single agent query from the CLI
python agent.py <owner>_<repo>_<mode> "your question here"

# Run the Streamlit UI
streamlit run app.py
```

No test runner, linter, or formatter is configured yet. If you add one, update this section.

## Architecture

The system is a from-scratch RAG agent — **no LangChain, no LlamaIndex**. Keep that constraint when adding features; do not introduce those frameworks.

Data flow: `ingest.py` pulls a GitHub repo via PyGithub, embeds chunks with Ollama (`nomic-embed-text`), and persists them to ChromaDB at `./chroma_db`. At query time, `app.py` (Streamlit) calls into `agent.py`, which runs an Anthropic tool-use loop against the Claude API. The agent reaches the index through tools defined in `tools.py` — the retrieval path is tool-call mediated, not a direct RAG pipeline, so prompts and tool schemas in `prompts.py` / `tools.py` are the primary surface for behavior changes.

Module responsibilities (per README):
- `ingest.py` — repo fetch + chunk + embed + write to Chroma. Runs standalone.
- `agent.py` — Anthropic SDK conversation loop (`claude-opus-4-7`) with tool dispatch; exposes `run_agent(user_query, collection_name)`.
- `tools.py` — tool definitions (schemas + implementations) the agent invokes; this is where vector search, file reads, etc. live. `TOOL_SCHEMAS` is in Claude's format (`{name, description, input_schema}`), not OpenAI's.
- `prompts.py` — `SYSTEM_PROMPT` and `QUERY_REWRITE_PROMPT` (the latter has a `{query}` placeholder for `.format()`).
- `logger.py` — structured JSONL logging to `agent_logs.jsonl` (gitignored).
- `app.py` — Streamlit front end wrapping the agent.
- `deploy/vllm_modal.py` — **vestigial.** Kept for reference only; the project no longer uses vLLM or any OpenAI-compatible endpoint.
- `eval/` — offline evaluation harness: `test_queries.py` (benchmark set), `metrics.py` (scoring), `compare.py` (run/config comparison). Writes `eval_results.jsonl` / `benchmark_results.json` (both gitignored).

## Model and dependencies

- LLM: **Claude API** (`claude-opus-4-7`) via the official `anthropic` Python SDK. Requires `ANTHROPIC_API_KEY` in `.env`. Opus 4.7 accepts only adaptive thinking (`thinking={"type": "adaptive"}`); `budget_tokens`, `temperature`, `top_p`, and `top_k` all return 400. `TOOL_SCHEMAS` must be in Claude's format.
- Embeddings: Ollama `nomic-embed-text`, accessed via the `ollama` Python client — embeddings are local, so the Ollama daemon must be running during ingest and query.
- Vector store: ChromaDB persistent client rooted at `./chroma_db` (gitignored). Treat the directory as disposable; rebuild via `python ingest.py`.

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `benchmark_results.json` are gitignored. Never commit them.
