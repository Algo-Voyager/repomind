# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

All Python modules (`agent.py`, `app.py`, `ingest.py`, `tools.py`, `prompts.py`, `logger.py`, `eval/*.py`) are currently empty scaffolds. The README defines the intended architecture; implementation has not started. When adding code, conform to the module responsibilities described below rather than inventing a new layout.

## Commands

Setup assumes a local venv and a running Ollama daemon with `nomic-embed-text` pulled.

```bash
# One-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
cp .env.example .env   # fill in ANTHROPIC_API_KEY and GITHUB_TOKEN

# Build the local vector index (writes to ./chroma_db)
python ingest.py

# Run the Streamlit UI
streamlit run app.py
```

No test runner, linter, or formatter is configured yet. If you add one, update this section.

## Architecture

The system is a from-scratch RAG agent — **no LangChain, no LlamaIndex**. Keep that constraint when adding features; do not introduce those frameworks.

Data flow: `ingest.py` pulls a GitHub repo via PyGithub, embeds chunks with Ollama (`nomic-embed-text`), and persists them to ChromaDB at `./chroma_db`. At query time, `app.py` (Streamlit) calls into `agent.py`, which runs a Claude tool-use loop. The agent reaches the index through tools defined in `tools.py` — the retrieval path is tool-call mediated, not a direct RAG pipeline, so prompts and tool schemas in `prompts.py` / `tools.py` are the primary surface for behavior changes.

Module responsibilities (per README):
- `ingest.py` — repo fetch + chunk + embed + write to Chroma. Runs standalone.
- `agent.py` — Claude conversation loop with tool dispatch.
- `tools.py` — tool definitions (schemas + implementations) the agent invokes; this is where vector search, file reads, etc. live.
- `prompts.py` — system and task prompts.
- `logger.py` — structured JSONL logging to `agent_logs.jsonl` (gitignored).
- `app.py` — Streamlit front end wrapping the agent.
- `eval/` — offline evaluation harness: `test_queries.py` (benchmark set), `metrics.py` (scoring), `compare.py` (run/config comparison). Writes `eval_results.jsonl` / `benchmark_results.json` (both gitignored).

## Model and dependencies

- LLM: `claude-opus-4-7` via the `anthropic` SDK. If updating models, check the Anthropic docs for the current identifier.
- Embeddings: Ollama `nomic-embed-text`, accessed via the `ollama` Python client — embeddings are local, so the Ollama daemon must be running during ingest and query.
- Vector store: ChromaDB persistent client rooted at `./chroma_db` (gitignored). Treat the directory as disposable; rebuild via `python ingest.py`.

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `benchmark_results.json` are gitignored. Never commit them.
