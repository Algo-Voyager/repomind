"""AST vs naive chunking benchmark — LLM-as-judge with Qwen.

For each benchmark query, retrieves the top-N chunks from the AST and
naive collections and asks Qwen to rate each chunk's relevance 1-5.
Writes ``frontend/public/benchmark_results.json`` (gitignored).

Usage:
    python eval/compare.py <owner>/<repo>

Embedding retrieval uses the same Modal EMBED_BASE_URL as ingest/tools.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

import time

import httpx
import openai
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL   = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_PATH   = "./chroma_db"
OUTPUT_PATH   = Path("frontend/public/benchmark_results.json")

QWEN_GENERATE_URL = os.getenv("QWEN_GENERATE_URL", "")
VLLM_API_KEY      = os.getenv("VLLM_API_KEY", "")

_embed: openai.OpenAI | None = None


def _get_embed() -> openai.OpenAI:
    global _embed
    if _embed is None:
        base_url = os.getenv("EMBED_BASE_URL")
        if not base_url:
            raise RuntimeError("EMBED_BASE_URL is not set. Add it to .env.")
        _embed = openai.OpenAI(
            base_url=base_url,
            api_key=VLLM_API_KEY,
        )
    return _embed


def _qwen(prompt: str, retries: int = 3) -> str:
    if not QWEN_GENERATE_URL:
        raise RuntimeError("QWEN_GENERATE_URL is not set. Add it to .env.")
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.post(
                QWEN_GENERATE_URL,
                json={"prompt": prompt, "max_new_tokens": 64, "temperature": 0.1},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                },
                timeout=120.0,
            )
            if resp.status_code >= 500:
                print(f"    [attempt {attempt}/{retries}] Qwen 500 — {'retrying in 10s…' if attempt < retries else 'giving up.'}")
                if attempt < retries:
                    time.sleep(10)
                    continue
                raise RuntimeError(f"Qwen service error 500 after {retries} attempts: {resp.text[:300]}")
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except httpx.TimeoutException:
            print(f"    [attempt {attempt}/{retries}] Timeout — {'retrying…' if attempt < retries else 'giving up.'}")
            if attempt == retries:
                raise
    return ""


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
    import chromadb
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma.get_collection(collection_name)

    resp = _get_embed().embeddings.create(model=EMBED_MODEL, input=query)
    embedding = resp.data[0].embedding

    results = col.query(
        query_embeddings=[embedding],
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

    text = _qwen(prompt)
    try:
        for ch in text:
            if ch in "12345":
                return int(ch)
        return 3
    except (ValueError, IndexError):
        return 3


def _print_chunks(label: str, chunks: list[dict], scores: list[int]) -> None:
    print(f"\n    ── {label} retrieved chunks ──")
    for i, (c, s) in enumerate(zip(chunks, scores), 1):
        meta = c["metadata"] or {}
        kind      = meta.get("type", "chunk")
        file_path = meta.get("file_path", "?")
        name      = meta.get("name", "")
        line_s    = meta.get("line_start", "")
        line_e    = meta.get("line_end", "")
        dist      = f"{c['distance']:.3f}"
        loc = f"{file_path}:{line_s}-{line_e}" if line_s else file_path
        header = f"[{i}] score={s}  dist={dist}  {kind}"
        header += f" `{name}`" if name else ""
        header += f"  @ {loc}"
        print(f"    {header}")
        # Show first 300 chars of the chunk text
        preview = c["text"].replace("\n", " ").strip()[:300]
        print(f"        {preview}")


def run_benchmark(repo_owner: str, repo_name: str, verbose: bool = False) -> dict:
    ast_col   = f"{repo_owner}_{repo_name}_ast"
    naive_col = f"{repo_owner}_{repo_name}_naive"

    print(f"Comparing  {ast_col}  vs  {naive_col}")
    if verbose:
        print("  (verbose: showing retrieved chunks and scores)\n")

    results: list[dict] = []
    for query in BENCHMARK_QUERIES:
        print(f"\n  query: {query}")

        ast_chunks   = retrieve(ast_col,   query)
        naive_chunks = retrieve(naive_col, query)

        ast_scores   = [score_chunk(c["text"], query) for c in ast_chunks]
        naive_scores = [score_chunk(c["text"], query) for c in naive_chunks]

        if verbose:
            _print_chunks("AST",   ast_chunks,   ast_scores)
            _print_chunks("Naive", naive_chunks, naive_scores)

        ast_avg   = round(mean(ast_scores),   2) if ast_scores   else 0.0
        naive_avg = round(mean(naive_scores), 2) if naive_scores else 0.0

        if ast_avg > naive_avg:
            winner = "ast"
        elif naive_avg > ast_avg:
            winner = "naive"
        else:
            winner = "tie"

        results.append({
            "query":           query,
            "ast_avg_score":   ast_avg,
            "naive_avg_score": naive_avg,
            "ast_scores":      ast_scores,
            "naive_scores":    naive_scores,
            "winner":          winner,
            "delta":           round(ast_avg - naive_avg, 2),
        })
        print(f"    AST {ast_avg:.1f}  |  Naive {naive_avg:.1f}  |  {winner.upper()} wins")

    ast_wins   = sum(1 for r in results if r["winner"] == "ast")
    naive_wins = sum(1 for r in results if r["winner"] == "naive")
    ties       = sum(1 for r in results if r["winner"] == "tie")

    summary = {
        "repo":             f"{repo_owner}/{repo_name}",
        "total_queries":    len(results),
        "ast_wins":         ast_wins,
        "naive_wins":       naive_wins,
        "ties":             ties,
        "avg_ast_score":    round(mean(r["ast_avg_score"]   for r in results), 2),
        "avg_naive_score":  round(mean(r["naive_avg_score"] for r in results), 2),
        "results":          results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  AST wins:   {ast_wins}/{len(results)}")
    print(f"  Naive wins: {naive_wins}/{len(results)}")
    print(f"  Ties:       {ties}/{len(results)}")
    print(f"  Avg AST:    {summary['avg_ast_score']}")
    print(f"  Avg Naive:  {summary['avg_naive_score']}")
    print(f"\n  Results written to {OUTPUT_PATH}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AST vs naive chunking benchmark")
    parser.add_argument("repo", help="owner/repo  e.g. Algo-Voyager/secure-lock-app")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each retrieved chunk and its score")
    args = parser.parse_args()
    if "/" not in args.repo:
        parser.error("repo must be in owner/repo format")
    owner, name = args.repo.split("/", 1)
    run_benchmark(owner, name, verbose=args.verbose)
