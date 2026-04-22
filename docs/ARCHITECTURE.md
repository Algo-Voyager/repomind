# Architecture

End-to-end working of repomind — a grounded code-Q&A agent over a GitHub repository.

## What it does

Given a GitHub repo and a natural-language question, repomind finds the relevant code via a local semantic index, then asks Claude to answer while *only* citing what it retrieved. Retrieval is mediated by tool calls, not fed into the prompt up front — the model decides what to fetch.

## High-level diagram

```
           ┌────────────────────────────────────────────────────────┐
           │                   User (Streamlit UI)                  │
           │                 or `python agent.py`                   │
           └───────────────┬────────────────────────────▲───────────┘
                           │ question                   │ answer
                           ▼                            │
                 ┌──────────────────┐                   │
                 │   agent.py       │  ReAct loop       │
                 │  (Claude API)    │  (tool_use ⇆      │
                 └───────┬──────────┘   tool_result)    │
                         │                              │
               ┌─────────┴──────────┐                   │
               │ query_rewrite      │                   │
               └─────────┬──────────┘                   │
                         │                              │
     Claude decides which tool to call:                 │
                         ▼                              │
  ┌────────────┬────────────────┬────────────────┐      │
  │vector_search│ get_file      │get_recent_      │     │
  │            │               │commits          │      │
  └────┬───────┴───────┬───────┴─────┬───────────┘      │
       │               │             │                  │
       ▼               ▼             ▼                  │
  ┌─────────┐    ┌───────────┐  ┌───────────┐           │
  │ Ollama  │    │  PyGithub │  │ PyGithub  │           │
  │ embed   │    │           │  │           │           │
  └────┬────┘    └─────┬─────┘  └─────┬─────┘           │
       │               │              │                 │
       ▼               ▼              ▼                 │
  ┌────────────┐   ┌──────────────────────┐             │
  │ ChromaDB   │   │       GitHub API     │             │
  │ (local)    │   │                      │             │
  └────┬───────┘   └──────────────────────┘             │
       │                                                │
       └──── retrieved chunks / files / commits ────────┘

  Side channels: logger.py → agent_logs.jsonl → eval/metrics.py
                                               → Streamlit Logs tab
```

## Components

| Module | Role |
|---|---|
| `ingest.py` | Fetch a repo via PyGithub, chunk files (AST for `.py`, H2 sections for `.md`, naive fixed-size otherwise), embed each chunk with Ollama `nomic-embed-text`, upsert into ChromaDB at `./chroma_db`. Collection name is `<owner>_<repo>_<mode>`. |
| `agent.py` | The orchestrator. Calls `query_rewrite` once, then loops `messages.create(...)` → inspect `stop_reason` → dispatch any `tool_use` blocks → feed `tool_result` back → repeat. Uses `claude-opus-4-7`. |
| `tools.py` | Three tools the agent calls: `vector_search` (semantic query against Chroma), `get_file` (raw file via PyGithub), `get_recent_commits`. `TOOL_SCHEMAS` is in Claude's format. `run_tool` is the dispatcher. |
| `prompts.py` | `SYSTEM_PROMPT` (agent rules + answer format) and `QUERY_REWRITE_PROMPT` (compresses the user question into a semantic-search query). |
| `logger.py` | Appends one JSON line per event to `agent_logs.jsonl`. `get_recent_logs` / `get_session_logs` are readers for the UI. |
| `app.py` | Streamlit UI. Three tabs: **Chat**, **Logs** (aggregate metrics + recent events), **Benchmarks** (AST vs naive + correctness pass rate). Sidebar ingests repos by shelling out to `python ingest.py`. |
| `eval/metrics.py` | Reads `agent_logs.jsonl` and computes per-session + aggregate stats: latency, tool/LLM counts, input/output tokens, estimated USD cost. |
| `eval/test_queries.py` | Correctness harness — runs 5 fixed queries through the agent and scores each by must-contain / must-not-contain keywords. Writes `eval_results.jsonl`. |
| `eval/compare.py` | AST vs naive benchmark — retrieves top-3 chunks from both collections for 8 queries, asks Claude to rate each chunk 1–5, writes `benchmark_results.json`. |
| `deploy/vllm_modal.py` | **Vestigial.** From an earlier iteration that used self-hosted vLLM. Kept in the tree but no longer wired in. |

## Workflow 1 — Ingestion (`python ingest.py owner/repo ast`)

1. Load `GITHUB_TOKEN` from `.env`, resolve the repo via PyGithub.
2. Walk the tree, skipping `node_modules` / `.git` / `dist` / `build` and anything over 500 KB.
3. For each allowed file (`.py`, `.md`, `.ts`, `.tsx`, `.js`, `.jsx`, `.txt`):
   - `.py` + `ast` mode → parse with `ast.walk`, emit one chunk per function / class.
   - `.md` → split at `## ` H2 headings, one chunk per section.
   - Anything else → naive fixed-size window (2000 chars, 50 overlap).
4. Embed each chunk via Ollama `nomic-embed-text` (local, no network call).
5. Upsert into Chroma collection `<owner>_<repo>_<mode>` with metadata: `type`, `name`, `file_path`, `line_start`, `line_end`, `language`, `heading`.

**Result:** a persistent Chroma collection at `./chroma_db` ready for retrieval.

## Workflow 2 — Query (`python agent.py <collection> "question"` or UI)

1. **Query rewrite.** `agent.query_rewrite` sends the user question to Claude with `QUERY_REWRITE_PROMPT` and gets back a compact semantic query (e.g. *"how do I log in?"* → *"authentication login flow implementation"*).
2. **Seed the conversation.** The first user message contains both the original and rewritten questions.
3. **ReAct loop** (up to `max_steps=6`):
   1. Call `client.messages.create(model="claude-opus-4-7", system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=...)`.
   2. If `stop_reason == "end_turn"` → extract final text, log it, return.
   3. If `stop_reason == "refusal"` → surface a canned refusal message, return.
   4. Otherwise pull every `tool_use` block, append the assistant turn verbatim (Claude requires this to match `tool_use_id` later), run each tool via `run_tool`, append a single user message containing one `tool_result` block per call, loop.
4. **Tool dispatch.** `run_tool(name, args, collection_name)` injects the collection into the kwargs and calls:
   - `vector_search` → embed the query, query Chroma with optional `where={"type": filter}`, format results as text for the model.
   - `get_file` → parse `owner_repo` from the collection name, fetch via PyGithub, return first 3000 chars.
   - `get_recent_commits` → PyGithub, last N commits, formatted.
5. **Logging.** Every LLM call, tool call, tool result, final answer (with token usage), refusal, or max-steps event is appended to `agent_logs.jsonl` via `logger.log_step`.
6. **Return.** The return payload includes `session_id`, `answer`, `steps`, `messages`, and `usage.input_tokens / output_tokens` — the UI renders the answer plus an expandable per-step trace from `get_session_logs(session_id)`.

## Workflow 3 — Evaluation

- **Correctness** (`eval/test_queries.py`) — drives the agent over 5 predefined questions and checks each answer for must-contain / must-not-contain keyword lists. Pass rate is printed; per-query rows go to `eval_results.jsonl`.
- **AST vs naive** (`eval/compare.py`) — retrieves top-3 chunks from `<owner>_<repo>_ast` and `<owner>_<repo>_naive` for each of 8 benchmark queries and asks Claude to score each chunk 1–5. Outputs `benchmark_results.json` with win counts, per-query deltas, and averages. Requires both collections ingested.
- **Aggregate metrics** (`eval/metrics.py`) — reads `agent_logs.jsonl`, computes per-session totals (latency, steps, tool counts, tokens) and an overall summary (avg / median / p95 latency, total tokens, total USD cost based on Opus 4.7 list price).

## Data stores

| File / directory | Written by | Read by | Gitignored |
|---|---|---|---|
| `./chroma_db/` | `ingest.py` | `tools.py::vector_search`, `eval/compare.py` | ✅ |
| `agent_logs.jsonl` | `logger.py` | `eval/metrics.py`, `app.py` (Logs tab) | ✅ |
| `eval_results.jsonl` | `eval/test_queries.py` | `app.py` (Benchmarks tab) | ✅ |
| `benchmark_results.json` | `eval/compare.py` | `app.py` (Benchmarks tab) | ✅ |
| `.env` | user | everything needing secrets | ✅ |

## Design choices worth knowing

- **Tool-call-mediated retrieval, not a fixed RAG pipeline.** The model decides when to search, when to read a whole file, and when to stop. `prompts.py` and the tool descriptions are the primary behavior-tuning surface.
- **Embeddings are local.** Ollama runs `nomic-embed-text` on the host — no embedding calls hit Anthropic or any other vendor. Only the generative LLM calls go to Claude.
- **Collections are versioned by chunking mode.** `<owner>_<repo>_ast` and `<owner>_<repo>_naive` live side by side; this is what makes the AST vs naive benchmark possible without re-ingesting.
- **`claude-opus-4-7` constraints.** Adaptive thinking only; no `temperature` / `top_p` / `top_k` / `budget_tokens` — all return 400. The agent deliberately avoids these parameters.
- **Prompt caching / adaptive thinking not enabled.** Trivial to turn on; omitted to keep the agent loop spec-faithful. Enabling adaptive thinking on the main loop would improve long-horizon reasoning at some extra cost.
