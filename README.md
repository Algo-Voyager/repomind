# repomind

A developer documentation agent: point it at a GitHub repo, and ask questions about the code. Answers are grounded in the repo — the agent retrieves chunks from a local vector index via tool calls and cites file paths + line numbers.

## Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` on Modal via rag-learning's `QwenService` (custom `/generate` endpoint) |
| **LLM client** | `httpx` — direct POST to `QWEN_GENERATE_URL` |
| **Agent loop** | Text-based ReAct (Thought → Action → Observation → Final Answer) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` on Modal via rag-learning's `embedding_api` — OpenAI-compatible `/v1/embeddings` |
| **Embed client** | `openai` Python SDK pointed at `EMBED_BASE_URL` |
| **Vector DB** | ChromaDB (persistent, stored in `./chroma_db`) |
| **GitHub API** | PyGithub |
| **UI** | Streamlit |
| **Background jobs** | [Inngest](https://www.inngest.com) via FastAPI — ingest runs as a 2-step durable job; agent runs are broken into per-step checkpoints |

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

### 4. Deploy rag-learning Modal services

Both the LLM and embeddings run on Modal under the `qwen-7b-service` app (shared with rag-learning). Deploy once from the rag-learning repo:

```bash
cd ../rag-learning
modal deploy qwen_modal.py
```

Modal prints two URLs:
- **LLM generate**: `https://<workspace>--qwen-7b-service-qwenservice-generate.modal.run`
- **Embeddings**: `https://<workspace>--qwen-7b-service-embedding-api.modal.run`

To stop and clean up all Modal resources:

```bash
bash cleanup_modal.sh
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `VLLM_API_KEY` | API key shared by both Modal services |
| `QWEN_GENERATE_URL` | LLM generate endpoint URL from Modal |
| `EMBED_BASE_URL` | Embedding endpoint URL from Modal **+ `/v1`** |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `GITHUB_TOKEN` | GitHub personal access token with repo read access |

> **Important:** `EMBED_BASE_URL` must end with `/v1` — the OpenAI SDK appends `/embeddings` to it, so the full path becomes `.../v1/embeddings`.

### 6. Start the FastAPI + Inngest backend

```bash
# Terminal 1 — FastAPI + Inngest server
uvicorn server:app --port 8000
```

### 7. Start the Inngest Dev Server

```bash
# Terminal 2 — Inngest Dev Server (UI at http://localhost:8288)
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

Open http://localhost:8288 to monitor ingest jobs and agent runs step-by-step.

### 8. Launch the Streamlit UI

```bash
# Terminal 3
streamlit run app.py
```

The UI has three tabs:
- **Chat** — ask the agent questions about an ingested repo
- **Logs** — recent activity, latency, and estimated cost
- **Benchmarks** — AST vs naive chunking quality + correctness pass rate

**Ingest a repo from the sidebar** — clicking "Ingest repo" triggers a background Inngest job (2 steps: `fetch-and-chunk` → `embed-and-store`). Watch progress at localhost:8288.

### CLI usage (no server required)

```bash
# Ingest directly (bypasses Inngest)
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

| Function | Trigger | Steps visible in Dev UI |
|----------|---------|------------------------|
| `repomind/ingest_repo` | Sidebar "Ingest repo" button | `fetch-and-chunk` → `embed-and-store` |
| `repomind/run_agent` | `POST /api/query` (async API path) | `query-rewrite` → `llm-generate-1` → `vector-search-1` → `llm-generate-2` → … |
| `repomind/agent_completed` | After every Streamlit chat query | `compute-metrics` (latency, embed_ms, chroma_ms) |

Each step shows its own duration in the timeline. LLM generate steps are also memoized — on retry, already-completed steps are not re-called.

### Event payloads

**`repomind/ingest_repo`** — triggered when you click "Ingest repo" in the sidebar:
```json
{
  "name": "repomind/ingest_repo",
  "data": {
    "repo": "GodsScion/Auto_job_applier_linkedIn",
    "mode": "ast"
  }
}
```

**`repomind/run_agent`** — triggered by `POST /api/query`:
```json
{
  "name": "repomind/run_agent",
  "data": {
    "query": "How does the login flow work?",
    "collection_name": "GodsScion_Auto_job_applier_linkedIn_ast"
  }
}
```

**`repomind/agent_completed`** — fired automatically after every Streamlit chat query:
```json
{
  "name": "repomind/agent_completed",
  "data": {
    "session_id": "abc123",
    "query": "How does the login flow work?",
    "latency_ms": 3200,
    "input_tokens": 1540,
    "output_tokens": 320
  }
}
```

## Architecture & Challenges

### How the agent works

```
User question
    │
    ▼
query-rewrite (Inngest step)       ← compact semantic search query
    │
    ▼
ReAct loop (up to 6 iterations):
    │
    ├─ llm-generate-N (Inngest step)   ← Qwen generates Thought + Action
    │       │
    │       ├─ "Final Answer:" found → return answer
    │       │
    │       └─ Action parsed → tool call
    │               │
    │               └─ vector-search-N / get-file-N (Inngest step)
    │                       │
    │                       └─ Observation appended to prompt
    ▼
Final Answer returned to Streamlit
```

### Challenges faced and how they were solved

**1. Embeddings: local Ollama → Modal**

Initially embeddings used Ollama running locally (`nomic-embed-text`). This required every developer to run a local daemon and meant the model never worked in cloud environments.

*Solution:* Deployed `BAAI/bge-small-en-v1.5` on Modal as an OpenAI-compatible `/v1/embeddings` endpoint using `sentence-transformers`. The `openai` SDK calls it the same way it would call any embedding API.

*Gotcha:* `EMBED_BASE_URL` must include `/v1` (e.g. `.../v1` not just `...`) because the OpenAI SDK constructs the full URL as `base_url + /embeddings`. Without `/v1`, the path becomes `.../embeddings` which returns 404.

---

**2. vLLM cold-start timeout**

The `repomind-vllm` Modal deployment kept failing. Qwen2.5-7B is ~14 GB of weights. On the first cold start (especially after volumes are deleted), vLLM needs to download and load all the weights before it can serve a request. The original `startup_timeout` was 10 minutes — exactly where the containers were dying.

*Solution:* Increased `startup_timeout` to 20 minutes and removed `@modal.concurrent` which is not valid for `@modal.web_server` functions. Ultimately switched to reusing the already-stable `qwen-7b-service` deployment from rag-learning, which had its weights already cached.

---

**3. OpenAI tool-calling → text-based ReAct**

The original agent used the OpenAI SDK's `tools=` parameter (function calling). This only works with OpenAI-compatible endpoints that support function calling (like vLLM). When switching to rag-learning's `QwenService`, the endpoint is a simple `POST /generate` that returns `{"response": "..."}` — no function calling support at all.

*Solution:* Rewrote the agent as a text-based ReAct loop. The prompt tells the model to output:
```
Thought: I need to search the codebase
Action: vector_search
Action Input: {"query": "authentication login flow"}
```
The agent parses `Action:` and `Action Input:` via regex, runs the tool, and appends `Observation: <result>` back to the prompt. The model reads the observation and continues until it writes `Final Answer:`.

---

**4. Inngest visibility — all one black box**

Originally the entire agent run was a single `agent-loop` Inngest step. You could see the total time but couldn't tell whether time was spent in the LLM, in vector search, or somewhere else.

*Solution:* Restructured `run_agent_fn` in `server.py` to drive the ReAct loop itself, making each LLM call and each tool call its own `ctx.step.run`. Now the Inngest timeline shows individual durations for `llm-generate-1`, `vector-search-1`, etc. Steps are also memoized, so retries don't re-call the LLM for already-completed work.

---

**5. embed_ms / chroma_ms always null in monitoring**

The `_TOOL_METRICS` ContextVar in `tools.py` correctly recorded embed and Chroma latency per tool call. But the monitoring event fired at the end of `run_agent` only included `session_id`, `steps`, and `total_latency_s` — the tool metrics were spread into `log_step` but never passed to `_fire_monitoring_event`.

*Solution:* Added `total_embed_ms` and `total_chroma_ms` accumulators in `run_agent`, summing up metrics across all tool calls in the run, and included them in every `_fire_monitoring_event` payload.

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
├── ingest.py              # Fetch repo, chunk (AST or naive), embed via Modal, write to Chroma
├── agent.py               # Text-based ReAct loop using httpx → QWEN_GENERATE_URL
├── tools.py               # Tool implementations (vector_search, get_file, get_recent_commits)
├── prompts.py             # REACT_PROMPT_TEMPLATE + QUERY_REWRITE_PROMPT
├── logger.py              # Structured JSONL logging to agent_logs.jsonl
├── app.py                 # Streamlit UI (Chat / Logs / Benchmarks)
├── cleanup_modal.sh       # Stop Modal app + delete cached volumes
├── deploy/
│   └── qwen_modal.py      # Modal deployment: vLLM (LLM) + sentence-transformers (embeddings)
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
