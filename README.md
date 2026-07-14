# KodeWiki

Ask plain-English questions about any GitHub repo and get answers grounded in the code with file-path + line-number citations. The agent runs vector search over chunks of the repo, then a ReAct loop to assemble the answer.

**Live:** [kodewiki.vercel.app](https://kodewiki.vercel.app)
**Architecture deep-dive:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Stack

| Layer | Default | Override per user |
|---|---|---|
| **Text generation** | Qwen2.5-7B on Modal (custom `/generate`) | OpenAI / Gemini via Settings → X-LLM-* headers |
| **Embeddings** | bge-small-en-v1.5 (384d) on Modal | OpenAI `text-embedding-3-small` (1536d) / Gemini `gemini-embedding-001` (3072d) |
| **Code chunking** | Python `ast` for `.py`; **tree-sitter cAST** for 306+ other languages; H2-headings for `.md`; naive sliding-window as fallback | — |
| **Agent loop** | Text-based ReAct (parses `Action:` / `Action Input:`) | — |
| **Vector DB** | ChromaDB persistent client (`./chroma_db`) | — |
| **Backend** | FastAPI + [Inngest](https://www.inngest.com) durable jobs | — |
| **Frontend** | Next.js (App Router) | — |
| **Production hosts** | Vercel + Render + Inngest Cloud + Modal | — |

No LangChain. No LlamaIndex.

> **Chunking:** AST-aware splits beat sliding-window for code RAG — the [cAST (2025)](https://arxiv.org/abs/2506.15655) paper reports +4.3 Recall@5 / +5.6 Pass@1. Python uses the stdlib `ast` module for richer metadata; everything else uses tree-sitter via [`tree_sitter_chunker.py`](tree_sitter_chunker.py) with cAST's recursive split-then-merge algorithm.

---

## Multi-tenant & BYO keys

Every browser gets an **anonymous tenant ID** (UUID in localStorage, sent as `X-Tenant-Id`). It scopes:

- Sidebar collections list
- Ingested ChromaDB collections (`{tenant}__{repo}_{mode}__{embed_provider}`)
- Agent logs + metrics
- Chunks endpoint, query results, chat history

Open `/settings` to override the deploy's defaults (stored in `localStorage`, never persisted on the server):

| Card | Headers | When to use |
|---|---|---|
| GitHub PAT | `X-Github-Token` | Server default expired / private repo access |
| Text Generation | `X-LLM-Provider` + `X-LLM-Key` + `X-LLM-Model` | Run on your OpenAI / Gemini key |
| Embeddings | `X-Embed-Provider` + `X-Embed-Key` + `X-Embed-Model` | Use your own embedding model (different dim = separate workspace) |
| Modal Bearer Key | `X-VLLM-Key` | Custom Modal deployment |
| Workspace Identity | `X-Tenant-Id` | Reset / import a tenant ID across devices |

---

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in VLLM_API_KEY, QWEN_GENERATE_URL, EMBED_BASE_URL, GITHUB_TOKEN

# (one-time) deploy Modal services
cd ../rag-learning && modal deploy qwen_modal.py && cd -

# 3 terminals
uvicorn server:app --reload --port 8000
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest          # UI at :8288
cd frontend && npm install && npm run dev                                  # UI at :3000
```

### Required env vars

| Variable | Description |
|---|---|
| `VLLM_API_KEY` | Modal Bearer key (shared for generate + embed endpoints) |
| `QWEN_GENERATE_URL` | LLM endpoint from Modal |
| `EMBED_BASE_URL` | Embedding endpoint from Modal — **must end with `/v1`** |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `GITHUB_TOKEN` | GitHub PAT (repo:read) |

### CLI (no server)

```bash
python ingest.py <owner>/<repo> <ast|naive>
python agent.py <owner>_<repo>_<mode> "How does error handling work?"
python tools.py <owner>_<repo>_<mode>     # smoke-test retrieval
```

---

## Usage

**Ingest** — paste `owner/repo` in the sidebar, pick AST or Naive, click *Ingest Repo*. The row appears instantly with a spinning ring around the chunk-count badge; live progress polls the backend every 3s. Hover a row for a 🗑 trash button (cancels in-flight ingests + drops the Chroma collection).

**Ask** — pick an indexed repo, type your question. The agent runs a ReAct loop (`vector_search` → `get_file` → `get_recent_commits` as needed). Mermaid diagrams render inline; chat history persists in `localStorage`.

**Inspect chunks** (any tenant):

```bash
BASE=https://<your-render>.onrender.com/api
TID=<your-tenant-id-from-settings>
curl "$BASE/collections" -H "X-Tenant-Id: $TID" | jq
curl "$BASE/collections/<name>/chunks?limit=5&chunk_type=function" -H "X-Tenant-Id: $TID" | jq
```

**Benchmark** (after ingesting both AST + Naive versions of one repo):

```bash
python eval/compare.py <owner>/<repo>     # results land in /benchmarks page
```

---

## Project layout

```
.
├── server.py              # FastAPI + Inngest functions, REST endpoints
├── auth.py                # ContextVars: tenant, GitHub PAT, LLM/embed provider+key+model
├── ingest.py              # Repo walk + Python AST chunker + dispatch
├── tree_sitter_chunker.py # cAST chunker for 306 languages
├── agent.py               # ReAct loop with provider switching (vllm/openai/gemini)
├── tools.py               # vector_search, get_file, get_recent_commits
├── prompts.py             # ReAct + query-rewrite + history-compression
├── logger.py              # JSONL logging tagged by tenant
├── inngest_setup.py       # Inngest client (prod via INNGEST_DEV=0)
├── render.yaml / DEPLOY.md
├── frontend/
│   ├── app/{chat,logs,benchmarks,settings}/
│   ├── components/Sidebar.tsx          # ingest UX + delete + tenant
│   ├── components/MarkdownRenderer.tsx # Mermaid sanitiser + sizing
│   └── lib/{api.ts,tenant.ts}          # auth headers + UUID
├── eval/{compare,test_queries,metrics}.py
└── docs/ARCHITECTURE.md
```

---

---

## Secrets and generated files

`.env`, `chroma_db/`, `agent_logs.jsonl`, `eval_results.jsonl`, `frontend/public/benchmark_results.json` are gitignored. Never commit them.
