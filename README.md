# repomind

A developer documentation agent powered by Claude, ChromaDB, and local embeddings via Ollama.

## Stack

- **LLM**: Claude (`claude-opus-4-7`) via the Anthropic API
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

- `ANTHROPIC_API_KEY` — your Anthropic API key
- `GITHUB_TOKEN` — a GitHub personal access token with repo read access

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
├── agent.py           # Claude agent loop
├── tools.py           # Tool definitions the agent can call
├── prompts.py         # System and task prompts
├── logger.py          # Structured logging to agent_logs.jsonl
├── app.py             # Streamlit UI
├── eval/
│   ├── compare.py         # Compare runs / configs
│   ├── test_queries.py    # Benchmark query set
│   └── metrics.py         # Evaluation metrics
├── requirements.txt
├── .env.example
└── .gitignore
```
