# repomind

A developer documentation agent: point it at a GitHub repo, and ask questions about the code. Answers are grounded in the repo — the agent retrieves chunks from a local vector index via tool calls and cites file paths + line numbers.

## Stack

- **LLM**: [Claude API](https://platform.claude.com) — `claude-opus-4-7` via the official `anthropic` Python SDK.
- **Embeddings**: `nomic-embed-text` running locally via [Ollama](https://ollama.com) — embeddings never leave the machine.
- **Vector DB**: ChromaDB (persistent, stored in `./chroma_db`).
- **GitHub API**: PyGithub.
- **UI**: Streamlit.

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
ollama serve   # usually runs automatically after install
```

The Ollama daemon must be running during both ingest and query — embeddings are computed locally.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `ANTHROPIC_API_KEY` — your Claude API key from https://console.anthropic.com.
- `GITHUB_TOKEN` — a GitHub personal access token with repo read access.

### 6. Ingest a repository

```bash
python ingest.py <owner>/<repo> <ast|naive>
```

- `ast` mode splits Python files at function / class boundaries and Markdown at H2 headings — the recommended mode.
- `naive` mode uses fixed-size character chunks — useful as a baseline for the AST vs naive benchmark.

This writes to a local ChromaDB collection named `<owner>_<repo>_<mode>` at `./chroma_db`.

### 7. Launch the UI

```bash
streamlit run app.py
```

The UI has three tabs: **Chat** (ask the agent), **Logs** (recent activity, latency, token usage, and cost), **Benchmarks** (AST vs naive ranking + correctness pass rate).

You can also drive the agent from the CLI:

```bash
python agent.py <owner>_<repo>_<mode> "How does error handling work?"
```

## Evaluation

Two harnesses live in `eval/`.

**Correctness** — runs a small fixed query set and checks each answer for must-contain / must-not-contain keywords:

```bash
python eval/test_queries.py <owner>_<repo>_<mode>
```

Writes `eval_results.jsonl` (gitignored) and prints a pass rate.

**AST vs naive benchmark** — retrieves top-N chunks from both modes for each of 8 benchmark queries and asks Claude (as judge) to rate each chunk 1–5:

```bash
python eval/compare.py <owner>/<repo>
```

Requires both `<owner>_<repo>_ast` and `<owner>_<repo>_naive` collections to be ingested. Writes `benchmark_results.json` (gitignored).

**Aggregate metrics** — prints totals across every session ever logged:

```bash
python eval/metrics.py
```

Reads `agent_logs.jsonl` and reports session counts, latency percentiles, token usage, and estimated Claude API spend.

## Project layout

```
repomind/
├── ingest.py          # Fetch repo, chunk (AST or naive), embed, write to Chroma
├── agent.py           # ReAct loop against the Claude API with tool dispatch
├── tools.py           # Tool definitions (vector_search, get_file, get_recent_commits)
├── prompts.py         # SYSTEM_PROMPT + QUERY_REWRITE_PROMPT
├── logger.py          # Structured JSONL logging to agent_logs.jsonl
├── app.py             # Streamlit UI (Chat / Logs / Benchmarks)
├── deploy/
│   └── vllm_modal.py      # Vestigial. Kept for reference only — not used.
├── eval/
│   ├── test_queries.py    # Correctness test suite (keyword match / anti-match)
│   ├── compare.py         # AST vs naive benchmark (Claude as judge)
│   └── metrics.py         # Per-session + aggregate metrics (latency, tokens, cost)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, and `benchmark_results.json` are gitignored. Never commit them.

## Pricing note

The cost numbers surfaced in the Logs tab and by `eval/metrics.py` are estimated from the `claude-opus-4-7` list price (**$5 / M input, $25 / M output**, cached 2026-04-15). If pricing changes, edit `INPUT_PRICE_PER_M` / `OUTPUT_PRICE_PER_M` at the top of `eval/metrics.py`.
