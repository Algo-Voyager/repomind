# repomind

A developer documentation agent: point it at a GitHub repo, and ask questions about the code. Answers are grounded in the repo — the agent retrieves chunks from a local vector index via tool calls and cites file paths + line numbers.

## Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Any OpenAI-compatible model via [vLLM](https://github.com/vllm-project/vllm) — default `Qwen/Qwen2.5-7B-Instruct` |
| **LLM client** | `openai` Python SDK with custom `base_url` (same pattern as rag-learning) |
| **Embeddings** | `nomic-embed-text` running locally via [Ollama](https://ollama.com) — never leaves the machine |
| **Vector DB** | ChromaDB (persistent, stored in `./chroma_db`) |
| **GitHub API** | PyGithub |
| **UI** | Streamlit |
| **Background jobs** | [Inngest](https://www.inngest.com) via FastAPI — repo ingestion runs as a durable 2-step job; every agent run fires a monitoring event |

No LangChain. No LlamaIndex. Everything is built from scratch.

## Setup

### 1. Clone and enter the project

```bash
cd repomind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama, then pull the embedding model

Install Ollama from https://ollama.com, then:

```bash
ollama pull nomic-embed-text
ollama serve   # usually starts automatically after install
```

The Ollama daemon must be running during both ingest and query — embeddings are computed locally.

### 5. Start a vLLM server

Repomind talks to any OpenAI-compatible endpoint. Start a local vLLM server (or point at any hosted OpenAI-compatible API):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
```

Or deploy to Modal (stays up, scales to zero when idle):

```bash
.venv/bin/modal deploy deploy/qwen_modal.py
# → sets VLLM_BASE_URL to the printed URL in .env

# To stop and clean up all Modal resources:
bash cleanup_modal.sh
```

### 6. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `VLLM_API_KEY` | API key for your vLLM server (can be any string for local servers) |
| `VLLM_BASE_URL` | Base URL of the vLLM OpenAI-compatible endpoint, e.g. `http://localhost:8001/v1` |
| `VLLM_MODEL` | Model name to use, e.g. `Qwen/Qwen2.5-7B-Instruct` |
| `GITHUB_TOKEN` | GitHub personal access token with repo read access |

### 7. Start the FastAPI + Inngest backend

The backend handles background ingestion jobs and agent run monitoring.

```bash
# Terminal 1 — FastAPI + Inngest server
uvicorn server:app --reload --port 8000
```

### 8. Start the Inngest Dev Server

```bash
# Terminal 2 — Inngest Dev Server (UI at http://localhost:8288)
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

Open http://localhost:8288 to monitor ingest jobs and agent runs step-by-step.

### 9. Launch the Streamlit UI

```bash
# Terminal 3
streamlit run app.py
```

The UI has three tabs:
- **Chat** — ask the agent questions about an ingested repo
- **Logs** — recent activity, latency, token usage, and estimated cost
- **Benchmarks** — AST vs naive chunking quality + correctness pass rate

**Ingest a repo from the sidebar** — clicking "Ingest repo" triggers a background Inngest job (2 steps visible in the Dev UI: `fetch-and-chunk` → `embed-and-store`). The Streamlit sidebar returns immediately; watch progress at localhost:8288.

### CLI usage (no server required)

```bash
# Ingest directly (bypasses Inngest, useful for scripting)
python ingest.py <owner>/<repo> <ast|naive>

# Single agent query
python agent.py <owner>_<repo>_<mode> "How does error handling work?"

# Smoke-test tools against an ingested collection
python tools.py <owner>_<repo>_<mode>
```

## PyCharm Run Configurations

The app has two Python processes. Create one debug configuration for each.

**Config 1 — FastAPI server (Inngest functions)**

Run → Edit Configurations → + → Python

| Field | Value |
|-------|-------|
| Name | `repomind-server` |
| Module name | `uvicorn` |
| Parameters | `server:app --port 8000` |
| Working dir | `/path/to/repomind` |
| Env vars | load from `.env` (use the "EnvFile" plugin or paste manually) |

> No `--reload` — reload spawns a child process the debugger loses track of.

**Config 2 — Streamlit UI (agent + tools)**

Run → Edit Configurations → + → Python

| Field | Value |
|-------|-------|
| Name | `repomind-streamlit` |
| Module name | `streamlit` |
| Parameters | `run app.py` |
| Working dir | `/path/to/repomind` |

Start the Inngest Dev Server separately in a terminal (it's Node.js, PyCharm can't debug it):

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

## Inngest monitoring

Three Inngest functions run in the background:

| Function | Trigger | Steps |
|----------|---------|-------|
| `repomind/ingest_repo` | Sidebar "Ingest repo" button | `fetch-and-chunk` → `embed-and-store` |
| `repomind/run_agent` | `POST /api/query` (async API path) | `agent-loop` |
| `repomind/agent_completed` | After every Streamlit chat query | `compute-metrics` (cost, latency, tokens) |

Inngest provides automatic retries, step-level error traces, and replay — visible at http://localhost:8288.

## Evaluation

Three harnesses live in `eval/`.

**Correctness** — runs a small fixed query set and checks each answer for must-contain / must-not-contain keywords:

```bash
python eval/test_queries.py <owner>_<repo>_<mode>
```

Writes `eval_results.jsonl` (gitignored) and prints a pass rate.

**AST vs naive benchmark** — retrieves top-N chunks from both modes for each of 8 benchmark queries and uses the LLM as judge to rate each chunk 1–5:

```bash
python eval/compare.py <owner>/<repo>
```

Requires both `<owner>_<repo>_ast` and `<owner>_<repo>_naive` collections to be ingested. Writes `benchmark_results.json` (gitignored).

**Aggregate metrics** — prints totals across every session ever logged:

```bash
python eval/metrics.py
```

Reads `agent_logs.jsonl` and reports session counts, latency percentiles, token usage, and estimated LLM spend.

## Project layout

```
repomind/
├── server.py              # FastAPI + Inngest server (3 functions, 3 REST endpoints)
├── inngest_setup.py       # Shared Inngest client singleton
├── ingest.py              # Fetch repo, chunk (AST or naive), embed, write to Chroma
├── agent.py               # ReAct loop via OpenAI SDK (vLLM-compatible) with tool dispatch
├── tools.py               # Tool definitions (vector_search, get_file, get_recent_commits)
├── prompts.py             # SYSTEM_PROMPT + QUERY_REWRITE_PROMPT
├── logger.py              # Structured JSONL logging to agent_logs.jsonl
├── app.py                 # Streamlit UI (Chat / Logs / Benchmarks)
├── cleanup_modal.sh       # Stop Modal app + delete cached volumes
├── deploy/
│   └── qwen_modal.py      # Modal deployment: Qwen2.5-7B via vLLM (OpenAI-compatible)
├── eval/
│   ├── test_queries.py    # Correctness test suite (keyword match / anti-match)
│   ├── compare.py         # AST vs naive benchmark (LLM as judge)
│   └── metrics.py         # Per-session + aggregate metrics (latency, tokens, cost)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `benchmark_results.json` are gitignored. Never commit them.

## Pricing note

Cost estimates in the Logs tab and `eval/metrics.py` default to **$0** for self-hosted/local models. To track spend for a hosted API, set in `.env`:

```
LLM_INPUT_PRICE_PER_M=0.50    # USD per 1M input tokens
LLM_OUTPUT_PRICE_PER_M=1.50   # USD per 1M output tokens
```
