"""AST vs naive chunking benchmark — LLM-as-judge with Claude.

For each benchmark query, retrieves the top-N chunks from the AST and
naive collections and asks Claude to rate each chunk's relevance 1-5.
Writes ``benchmark_results.json`` (gitignored).

Usage:
    python eval/compare.py <owner>/<repo>

Embedding retrieval uses the same Ollama ``nomic-embed-text`` model as
ingest; only the judge is on the Claude API.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

import anthropic
import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "claude-opus-4-7"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


BENCHMARK_QUERIES = [
    "how does authentication work",
    "what does the main function do",
    "where is error handling implemented",
    "how are configuration values loaded",
    "what functions handle data validation",
    "explain the class structure",
    "how is logging set up",
    "what does the API entry point do",
]


def retrieve(collection_name: str, query: str, n: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(collection_name)
    emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    results = col.query(
        query_embeddings=[emb["embedding"]],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": t, "metadata": m, "distance": d}
        for t, m, d in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def score_chunk(chunk_text: str, query: str) -> int:
    """LLM-as-judge: score 1-5 how relevant this chunk is to the query."""
    prompt = f"""Rate how relevant this code chunk is for answering the query.

Query: {query}

Chunk:
{chunk_text[:800]}

Scoring:
5 = perfectly relevant, complete self-contained unit
4 = highly relevant, minor context missing
3 = somewhat relevant but incomplete
2 = tangentially related
1 = irrelevant or cut-off mid-logic

Respond with ONLY a single digit 1-5. Nothing else."""

    response = _get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return int(text[0])
    except (ValueError, IndexError):
        return 3


def run_benchmark(repo_owner: str, repo_name: str) -> dict:
    ast_col = f"{repo_owner}_{repo_name}_ast"
    naive_col = f"{repo_owner}_{repo_name}_naive"

    results: list[dict] = []
    for query in BENCHMARK_QUERIES:
        print(f"\n🔍 {query}")

        ast_chunks = retrieve(ast_col, query)
        naive_chunks = retrieve(naive_col, query)

        ast_scores = [score_chunk(c["text"], query) for c in ast_chunks]
        naive_scores = [score_chunk(c["text"], query) for c in naive_chunks]

        ast_avg = mean(ast_scores) if ast_scores else 0.0
        naive_avg = mean(naive_scores) if naive_scores else 0.0

        if ast_avg > naive_avg:
            winner = "AST"
        elif naive_avg > ast_avg:
            winner = "Naive"
        else:
            winner = "Tie"

        results.append({
            "query": query,
            "ast_avg_score": round(ast_avg, 2),
            "naive_avg_score": round(naive_avg, 2),
            "ast_scores": ast_scores,
            "naive_scores": naive_scores,
            "winner": winner,
            "delta": round(ast_avg - naive_avg, 2),
        })

        print(
            f"   AST: {ast_avg:.1f} | Naive: {naive_avg:.1f} | "
            f"Winner: {winner} (+{abs(results[-1]['delta'])})"
        )

    ast_wins = sum(1 for r in results if r["winner"] == "AST")
    naive_wins = sum(1 for r in results if r["winner"] == "Naive")
    ties = sum(1 for r in results if r["winner"] == "Tie")

    summary = {
        "total_queries": len(results),
        "ast_wins": ast_wins,
        "naive_wins": naive_wins,
        "ties": ties,
        "avg_ast_score": round(mean(r["ast_avg_score"] for r in results), 2),
        "avg_naive_score": round(mean(r["naive_avg_score"] for r in results), 2),
        "results": results,
    }

    Path("benchmark_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"   AST wins:   {ast_wins}/{len(results)}")
    print(f"   Naive wins: {naive_wins}/{len(results)}")
    print(f"   Ties:       {ties}/{len(results)}")
    print(f"   Avg AST score:   {summary['avg_ast_score']}")
    print(f"   Avg Naive score: {summary['avg_naive_score']}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print("Usage: python eval/compare.py <owner>/<repo>", file=sys.stderr)
        sys.exit(2)
    owner, name = sys.argv[1].split("/", 1)
    run_benchmark(owner, name)
