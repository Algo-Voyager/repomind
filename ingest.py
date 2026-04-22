"""Ingest a GitHub repo into ChromaDB with AST-based or naive chunking.

Usage:
    python ingest.py <owner/name> <ast|naive>
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import chromadb
import ollama
from dotenv import load_dotenv
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


def embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a GitHub repo into ChromaDB.")
    parser.add_argument("repo", help='Repo in "owner/name" form, e.g. facebook/react')
    parser.add_argument("mode", choices=["ast", "naive"], help="Chunking mode")
    args = parser.parse_args()

    if args.repo.count("/") != 1:
        print(f"Repo must be in 'owner/name' form; got {args.repo!r}", file=sys.stderr)
        return 2
    owner, name = args.repo.split("/", 1)

    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set. Add it to .env.", file=sys.stderr)
        return 2

    gh = Github(auth=Auth.Token(token))
    try:
        repo = gh.get_repo(args.repo)
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

    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = f"{owner}_{name}_{args.mode}"
    collection = client.get_or_create_collection(collection_name)

    total = 0
    files_seen = 0
    for entry in walk_repo(gh, repo):
        if os.path.splitext(entry.path)[1] not in ALLOWED_EXTS:
            continue
        data = fetch_file_bytes(gh, entry)
        if data is None or is_binary(data):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue

        files_seen += 1
        chunks = chunk_file(entry.path, text, args.mode)
        if not chunks:
            continue

        ids: list[str] = []
        docs: list[str] = []
        vecs: list[list[float]] = []
        metas: list[dict] = []
        for chunk in chunks:
            try:
                vec = embed(chunk.text)
            except Exception as e:
                print(f"[warn] Embedding failed for {chunk.chunk_id}: {e}", flush=True)
                continue
            ids.append(chunk.chunk_id)
            docs.append(chunk.text)
            vecs.append(vec)
            metas.append(chunk.metadata)
            total += 1
            if total % 10 == 0:
                print(f"[progress] {total} chunks ingested (last: {entry.path})", flush=True)
        if ids:
            collection.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)

    print(
        f"\nDone. Ingested {total} chunks from {files_seen} files "
        f"into collection '{collection_name}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
