"""Agent tools: vector search, file fetch, recent commits.

Each tool returns a string formatted for the LLM to read. The agent
reaches the ingested index and the live GitHub repo through these
tools — all retrieval is tool-call mediated (see CLAUDE.md).

Usage (smoke test):
    python tools.py <collection_name>
"""

from __future__ import annotations

import json
import os
import sys
from itertools import islice

import chromadb
import ollama
from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
MAX_FILE_CHARS = 3000
VALID_FILTER_TYPES = {"function", "class", "doc", "code"}


def _parse_owner_repo(collection_name: str) -> tuple[str, str]:
    """Collection names are f'{owner}_{name}_{mode}' — strip mode, then split."""
    without_mode = collection_name.rsplit("_", 1)[0]
    owner, _, repo = without_mode.partition("_")
    if not owner or not repo:
        raise ValueError(
            f"Cannot parse owner/repo from collection_name {collection_name!r}"
        )
    return owner, repo


def _get_repo(collection_name: str):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set. Add it to .env.")
    owner, repo = _parse_owner_repo(collection_name)
    gh = Github(auth=Auth.Token(token))
    return gh.get_repo(f"{owner}/{repo}")


def _embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def vector_search(
    query: str,
    collection_name: str,
    filter_type: str | None = None,
    n_results: int = 5,
) -> str:
    """Semantic search over the ingested collection. Use FIRST for code/doc questions."""
    if filter_type is not None and filter_type not in VALID_FILTER_TYPES:
        return (
            f"Error: filter_type must be one of {sorted(VALID_FILTER_TYPES)}; "
            f"got {filter_type!r}."
        )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        return f"Error: collection {collection_name!r} not found ({e})."

    kwargs: dict = {
        "query_embeddings": [_embed(query)],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_type:
        kwargs["where"] = {"type": filter_type}

    results = collection.query(**kwargs)
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not ids:
        return f"No results for query {query!r}."

    lines = [f"Found {len(ids)} results:", ""]
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        meta = meta or {}
        kind = meta.get("type", "chunk")
        file_path = meta.get("file_path", "?")
        name = meta.get("name")
        line_start = meta.get("line_start")
        line_end = meta.get("line_end")
        heading = meta.get("heading")

        if kind in ("function", "class") and name and line_start and line_end:
            header = (
                f"[{i}] {kind} `{name}` in {file_path} "
                f"(lines {line_start}-{line_end})"
            )
        elif kind == "doc" and heading:
            header = f"[{i}] doc '{heading}' in {file_path}"
        else:
            header = f"[{i}] {kind} in {file_path}"

        lines.append(header)
        lines.append(doc)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def get_file(collection_name: str, file_path: str) -> str:
    """Fetch raw file content from the repo tied to this collection."""
    try:
        repo = _get_repo(collection_name)
    except (ValueError, RuntimeError) as e:
        return f"Error: {e}"

    try:
        entry = repo.get_contents(file_path)
    except Exception as e:
        return f"Error: could not fetch {file_path!r}: {e}"

    if isinstance(entry, list):
        return f"Error: {file_path!r} is a directory, not a file."

    try:
        text = entry.decoded_content.decode("utf-8")
    except (UnicodeDecodeError, AssertionError) as e:
        return f"Error: could not decode {file_path!r}: {e}"

    header = f"File: {file_path}\n{'=' * (6 + len(file_path))}\n"
    if len(text) > MAX_FILE_CHARS:
        return header + text[:MAX_FILE_CHARS] + "\n... [truncated]"
    return header + text


def get_recent_commits(collection_name: str, n: int = 5) -> str:
    """Return the last `n` commits of the repo tied to this collection."""
    try:
        repo = _get_repo(collection_name)
    except (ValueError, RuntimeError) as e:
        return f"Error: {e}"

    try:
        commits = list(islice(repo.get_commits(), n))
    except Exception as e:
        return f"Error: could not fetch commits: {e}"

    if not commits:
        return "No commits found."

    lines = [f"Last {len(commits)} commits:"]
    for c in commits:
        sha = c.sha[:7]
        author = c.commit.author.name if c.commit.author else "unknown"
        date = (
            c.commit.author.date.strftime("%Y-%m-%d")
            if c.commit.author and c.commit.author.date
            else "?"
        )
        subject = c.commit.message.splitlines()[0] if c.commit.message else ""
        lines.append(f"- {sha} ({date} by {author}): {subject}")
    return "\n".join(lines)


TOOL_SCHEMAS = [
    {
        "name": "vector_search",
        "description": (
            "Search the indexed codebase semantically. Use this FIRST "
            "for any question about code or docs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "filter_type": {
                    "type": "string",
                    "enum": ["function", "class", "doc", "code"],
                    "description": "Optional filter on chunk type",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file",
        "description": (
            "Fetch the raw content of a file from the GitHub repo. "
            "Use when you need the full file, not just a chunk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. 'src/main.py'",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "get_recent_commits",
        "description": (
            "Fetch the most recent commits from the GitHub repo. "
            "Use for questions about recent changes or history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of commits to fetch",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
]


_TOOL_DISPATCH = {
    "vector_search": vector_search,
    "get_file": get_file,
    "get_recent_commits": get_recent_commits,
}


def run_tool(tool_name: str, args: dict, collection_name: str) -> str:
    """Dispatch a tool call from the agent loop. Returns a string result."""
    fn = _TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return f"Error: unknown tool {tool_name!r}."
    try:
        return fn(collection_name=collection_name, **args)
    except TypeError as e:
        return f"Error: bad arguments to {tool_name}: {e}"


def _smoke_test(collection_name: str) -> None:
    print(f"=== vector_search('main entry point', {collection_name!r}) ===")
    print(vector_search("main entry point", collection_name))

    print(f"\n=== vector_search(..., filter_type='function') ===")
    print(vector_search("initialize the client", collection_name, filter_type="function"))

    print(f"\n=== get_recent_commits({collection_name!r}, n=3) ===")
    print(get_recent_commits(collection_name, n=3))

    print(f"\n=== run_tool('get_recent_commits', {{'n': 2}}, ...) ===")
    print(run_tool("get_recent_commits", {"n": 2}, collection_name))

    print(f"\n=== TOOL_SCHEMAS ({len(TOOL_SCHEMAS)} tools) ===")
    print(json.dumps([s["function"]["name"] for s in TOOL_SCHEMAS], indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools.py <collection_name>", file=sys.stderr)
        sys.exit(2)
    _smoke_test(sys.argv[1])
