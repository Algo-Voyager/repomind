# repomind

A developer documentation agent powered by a self-hosted vLLM server on Modal, ChromaDB, and local embeddings via Ollama.

## Stack

- **LLM**: Any HF model served by [vLLM](https://github.com/vllm-project/vllm) on [Modal](https://modal.com) (OpenAI-compatible API). Default: `Qwen/Qwen2.5-7B-Instruct`. See `deploy/vllm_modal.py`.
- **LLM client**: the `openai` Python SDK pointed at the Modal endpoint (no Anthropic / OpenAI hosted keys needed).
- **Embeddings**: `nomic-embed-text` running locally via [Ollama](https://ollama.com)
- **Vector DB**: ChromaDB (persistent, stored in `./chroma_db`)
- **GitHub API**: PyGithub
- **UI**: Streamlit

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

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `VLLM_BASE_URL` — your Modal vLLM endpoint (e.g. `https://<workspace>--repomind-vllm-serve.modal.run/v1`). Must end in `/v1`.
- `VLLM_API_KEY` — the API key you set when creating the `vllm-api-key` Modal secret
- `VLLM_MODEL` — the model ID vLLM is serving (must match `--served-model-name`)
- `GITHUB_TOKEN` — a GitHub personal access token with repo read access

See `deploy/vllm_modal.py` for the Modal deployment script (`modal serve deploy/vllm_modal.py` for dev, `modal deploy …` for persistent).

### 6. Ingest a repository

```bash
python ingest.py
```

This builds the local ChromaDB index at `./chroma_db`.

### 7. Launch the UI

```bash
streamlit run app.py
```

## Project layout

```
repomind/
├── ingest.py          # Pulls repo content and populates ChromaDB
├── agent.py           # Agent loop talking to the Modal vLLM server
├── tools.py           # Tool definitions the agent can call
├── prompts.py         # System and task prompts
├── logger.py          # Structured logging to agent_logs.jsonl
├── app.py             # Streamlit UI
├── deploy/
│   └── vllm_modal.py      # Modal deployment for the vLLM server
├── eval/
│   ├── compare.py         # Compare runs / configs
│   ├── test_queries.py    # Benchmark query set
│   └── metrics.py         # Evaluation metrics
├── requirements.txt
├── .env.example
└── .gitignore
```
