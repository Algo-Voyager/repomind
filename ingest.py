"""Ingest a GitHub repo into ChromaDB with AST-based or naive chunking.

Public API (used by Inngest steps in server.py):
    fetch_and_chunk_repo(repo_slug, mode, event_id) -> dict
    embed_and_store_chunks(chunks_data)             -> dict

CLI usage (calls both functions sequentially):
    python ingest.py <owner/name> <ast|naive>
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import openai
from dotenv import load_dotenv

load_dotenv()
from github import Auth, Github, GithubException, RateLimitExceededException
from github.ContentFile import ContentFile
from github.Repository import Repository


SKIP_DIRS = {"node_modules", ".git", "dist", "build"}
ALLOWED_EXTS = {".py", ".md", ".txt", ".js", ".ts", ".tsx", ".jsx"}
MAX_FILE_BYTES = 500 * 1024
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 50

EXT_LANGUAGE = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
}

# Match an H2 heading only (## , but not ### or deeper).
H2_PATTERN = re.compile(r"^##(?!#)\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def is_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def should_skip_path(path: str) -> bool:
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


def extract_python_ast_chunks(source: str, file_path: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_type = "function"
        elif isinstance(node, ast.ClassDef):
            node_type = "class"
        else:
            continue
        if node.end_lineno is None:
            continue
        text = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        docstring = ast.get_docstring(node) or ""
        chunks.append(
            Chunk(
                chunk_id=f"{file_path}::{node_type}::{node.name}::{node.lineno}",
                text=text,
                metadata={
                    "type": node_type,
                    "name": node.name,
                    "file_path": file_path,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "docstring": docstring,
                    "language": "python",
                },
            )
        )
    return chunks


def extract_markdown_chunks(source: str, file_path: str) -> list[Chunk]:
    matches = list(H2_PATTERN.finditer(source))
    if not matches:
        return [
            Chunk(
                chunk_id=f"{file_path}::doc::0",
                text=source,
                metadata={
                    "type": "doc",
                    "file_path": file_path,
                    "heading": "",
                    "language": "markdown",
                },
            )
        ]

    chunks: list[Chunk] = []
    first_start = matches[0].start()
    if first_start > 0 and source[:first_start].strip():
        chunks.append(
            Chunk(
                chunk_id=f"{file_path}::doc::preamble",
                text=source[:first_start].strip(),
                metadata={
                    "type": "doc",
                    "file_path": file_path,
                    "heading": "",
                    "language": "markdown",
                },
            )
        )
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        chunks.append(
            Chunk(
                chunk_id=f"{file_path}::doc::{i}",
                text=source[start:end].rstrip(),
                metadata={
                    "type": "doc",
                    "file_path": file_path,
                    "heading": m.group(1).strip(),
                    "language": "markdown",
                },
            )
        )
    return chunks


def naive_chunk(source: str, file_path: str, language: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = CHUNK_CHARS - CHUNK_OVERLAP
    idx = 0
    i = 0
    while i < len(source):
        text = source[i : i + CHUNK_CHARS]
        if text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{file_path}::code::{idx}",
                    text=text,
                    metadata={
                        "type": "code",
                        "file_path": file_path,
                        "chunk_index": idx,
                        "language": language,
                    },
                )
            )
            idx += 1
        i += step
    return chunks


def chunk_file(path: str, content: str, mode: str) -> list[Chunk]:
    ext = os.path.splitext(path)[1]
    language = EXT_LANGUAGE.get(ext, "text")
    if ext == ".py" and mode == "ast":
        chunks = extract_python_ast_chunks(content, path)
        if chunks:
            return chunks
        return naive_chunk(content, path, "python")
    if ext == ".md":
        return extract_markdown_chunks(content, path)
    return naive_chunk(content, path, language)


def wait_for_rate_limit(gh: Github) -> None:
    rl = gh.get_rate_limit().core
    wait = max(int(rl.reset.timestamp() - time.time()) + 1, 1)
    print(f"[rate-limit] GitHub API exhausted. Waiting {wait}s until reset...", flush=True)
    time.sleep(wait)


def walk_repo(gh: Github, repo: Repository) -> Iterable[ContentFile]:
    stack: list[str] = [""]
    while stack:
        path = stack.pop()
        entries: list[ContentFile] | ContentFile
        while True:
            try:
                entries = repo.get_contents(path)
                break
            except RateLimitExceededException:
                wait_for_rate_limit(gh)
            except GithubException as e:
                print(f"[warn] Skipping {path!r}: {e}", flush=True)
                entries = []
                break
        if isinstance(entries, ContentFile):
            entries = [entries]
        for entry in entries:
            if should_skip_path(entry.path):
                continue
            if entry.type == "dir":
                stack.append(entry.path)
            elif entry.type == "file":
                yield entry


def fetch_file_bytes(gh: Github, entry: ContentFile) -> bytes | None:
    if entry.size and entry.size > MAX_FILE_BYTES:
        return None
    while True:
        try:
            return entry.decoded_content
        except RateLimitExceededException:
            wait_for_rate_limit(gh)
        except (GithubException, AssertionError) as e:
            print(f"[warn] Could not read {entry.path}: {e}", flush=True)
            return None


_embed_client: openai.OpenAI | None = None


def _get_embed_client() -> openai.OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = openai.OpenAI(
            base_url=os.getenv("EMBED_BASE_URL"),
            api_key=os.getenv("VLLM_API_KEY", ""),
        )
    return _embed_client


def embed(text: str) -> list[float]:
    model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    resp = _get_embed_client().embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


def fetch_and_chunk_repo(repo_slug: str, mode: str, event_id: str = "cli") -> dict:
    """Step 1: Walk a GitHub repo, chunk every file, write to a temp JSONL on disk.

    The temp file is named with *event_id* so Inngest retries are idempotent —
    a retry of this step will overwrite the same file rather than producing a
    duplicate.

    Returns a small dict (safe to serialise as an Inngest step result):
        {collection_name, temp_path, files_seen, total_chunks}
    """
    load_dotenv()
    if repo_slug.count("/") != 1:
        raise ValueError(f"repo must be 'owner/name'; got {repo_slug!r}")
    owner, name = repo_slug.split("/", 1)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set. Add it to .env.")

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_slug)

    collection_name = f"{owner}_{name}_{mode}"
    safe_event_id = event_id.replace("/", "-")[:40]
    temp_path = str(Path("./chroma_db") / f".chunks_{collection_name}_{safe_event_id}.jsonl")
    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)

    files_seen = 0
    total_chunks = 0
    with open(temp_path, "w", encoding="utf-8") as f:
        for entry in walk_repo(gh, repo):
            if os.path.splitext(entry.path)[1] not in ALLOWED_EXTS:
                continue
            raw = fetch_file_bytes(gh, entry)
            if raw is None or is_binary(raw):
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files_seen += 1
            for chunk in chunk_file(entry.path, text, mode):
                f.write(json.dumps({
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }) + "\n")
                total_chunks += 1
            if total_chunks % 50 == 0 and total_chunks:
                print(f"[progress] {total_chunks} chunks chunked (last: {entry.path})", flush=True)

    print(f"[fetch-and-chunk] {total_chunks} chunks from {files_seen} files → {temp_path}", flush=True)
    return {
        "collection_name": collection_name,
        "temp_path": temp_path,
        "files_seen": files_seen,
        "total_chunks": total_chunks,
    }


def embed_and_store_chunks(chunks_data: dict) -> dict:
    """Step 2: Read the temp JSONL from fetch_and_chunk_repo, embed, upsert ChromaDB.

    The temp file is cleaned up in a finally block so a failed/retried step 2
    can still find its input on retry (the file is only deleted on success).
    """
    collection_name: str = chunks_data["collection_name"]
    temp_path: str = chunks_data["temp_path"]

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(collection_name)

    total = 0
    errors = 0
    try:
        with open(temp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                try:
                    vec = embed(chunk["text"])
                except Exception as e:
                    print(f"[warn] Embedding failed for {chunk['id']}: {e}", flush=True)
                    errors += 1
                    continue
                collection.upsert(
                    ids=[chunk["id"]],
                    documents=[chunk["text"]],
                    embeddings=[vec],
                    metadatas=[chunk["metadata"]],
                )
                total += 1
                if total % 50 == 0:
                    print(f"[progress] {total} chunks embedded", flush=True)
    finally:
        # Only delete if the file exists — safe for retries (retry re-runs
        # fetch-and-chunk first, which recreates the file before this step).
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass

    print(
        f"[embed-and-store] Done. {total} chunks stored, {errors} errors → '{collection_name}'",
        flush=True,
    )
    return {
        "collection_name": collection_name,
        "total_chunks": total,
        "embed_errors": errors,
        "files_seen": chunks_data.get("files_seen", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a GitHub repo into ChromaDB.")
    parser.add_argument("repo", help='Repo in "owner/name" form, e.g. facebook/react')
    parser.add_argument("mode", choices=["ast", "naive"], help="Chunking mode")
    args = parser.parse_args()

    try:
        chunks_data = fetch_and_chunk_repo(args.repo, args.mode, event_id="cli")
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 2
    except GithubException as e:
        if e.status == 404:
            print(
                f"Repo {args.repo!r} returned 404. Common causes:\n"
                f"  1. Typo in the repo name (case-sensitive).\n"
                f"  2. Repo is private and your GITHUB_TOKEN doesn't grant access "
                f"to it. Fine-grained PATs require you to explicitly list the "
                f"repos they can read — check https://github.com/settings/"
                f"personal-access-tokens and add this repo (or use a classic "
                f"PAT with the 'repo' scope).\n"
                f"  3. Repo was deleted or you're on the wrong owner/org.\n"
                f"Verify with: curl -H 'Authorization: Bearer $GITHUB_TOKEN' "
                f"https://api.github.com/repos/{args.repo}",
                file=sys.stderr,
            )
        elif e.status == 401:
            print(
                "GITHUB_TOKEN was rejected (401). Check it hasn't expired, "
                "and that the token string in .env is complete.",
                file=sys.stderr,
            )
        else:
            print(f"Failed to open repo {args.repo!r}: {e}", file=sys.stderr)
        return 1

    result = embed_and_store_chunks(chunks_data)
    print(
        f"\nDone. Ingested {result['total_chunks']} chunks from "
        f"{result['files_seen']} files into collection '{result['collection_name']}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
