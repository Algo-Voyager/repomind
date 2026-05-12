"""Dump and inspect chunks stored in a ChromaDB collection.

Usage:
    # List all chunks (metadata only, no text)
    python eval/inspect_chunks.py Algo-Voyager/secure-lock-app ast

    # Show full text of each chunk
    python eval/inspect_chunks.py Algo-Voyager/secure-lock-app ast --text

    # Filter to a specific chunk type: function | class | doc | code
    python eval/inspect_chunks.py Algo-Voyager/secure-lock-app ast --type function

    # Filter to a specific file
    python eval/inspect_chunks.py Algo-Voyager/secure-lock-app ast --file lib/auth.dart

    # Limit how many chunks are shown
    python eval/inspect_chunks.py Algo-Voyager/secure-lock-app ast --limit 20
"""

from __future__ import annotations

import argparse
import textwrap

import chromadb

CHROMA_PATH = "./chroma_db"
SEP = "─" * 72


def inspect(
    collection_name: str,
    show_text: bool,
    filter_type: str | None,
    filter_file: str | None,
    limit: int,
) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: collection '{collection_name}' not found — {e}")
        return

    total = col.count()
    print(f"Collection : {collection_name}")
    print(f"Total chunks: {total}\n")

    where: dict | None = None
    if filter_type:
        where = {"type": filter_type}

    results = col.get(
        limit=limit,
        where=where,
        include=["documents", "metadatas"],
    )

    ids       = results.get("ids", [])
    docs      = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    shown = 0
    for chunk_id, doc, meta in zip(ids, docs, metadatas):
        meta = meta or {}
        if filter_file and filter_file not in meta.get("file_path", ""):
            continue

        kind      = meta.get("type", "?")
        file_path = meta.get("file_path", "?")
        name      = meta.get("name", "")
        line_s    = meta.get("line_start", "")
        line_e    = meta.get("line_end", "")
        part      = meta.get("part")
        docstring = meta.get("docstring", "")

        print(SEP)
        loc = f"{file_path}  lines {line_s}–{line_e}" if line_s else file_path
        part_note = f"  [part {part}]" if part is not None else ""
        print(f"  [{shown + 1}]  type={kind}{part_note}")
        if name:
            print(f"  name      : {name}")
        print(f"  location  : {loc}")
        print(f"  chunk_id  : {chunk_id}")
        if docstring:
            print(f"  docstring : {docstring[:120]}")
        if show_text:
            print(f"  text ({len(doc)} chars):")
            for line in textwrap.wrap(doc, width=80):
                print(f"    {line}")
        else:
            preview = doc.replace("\n", " ").strip()[:200]
            print(f"  preview   : {preview}")

        shown += 1

    print(SEP)
    print(f"\nShowing {shown} of {total} chunks  (use --limit N to see more)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB chunks")
    parser.add_argument("repo", help="owner/repo  e.g. Algo-Voyager/secure-lock-app")
    parser.add_argument("mode", choices=["ast", "naive"], help="Chunking mode")
    parser.add_argument("--text",  action="store_true", help="Show full chunk text")
    parser.add_argument("--type",  dest="filter_type",
                        choices=["function", "class", "doc", "code"],
                        help="Filter to a specific chunk type")
    parser.add_argument("--file",  dest="filter_file",
                        help="Filter to chunks from a specific file path (substring match)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max chunks to fetch (default 50)")
    args = parser.parse_args()

    if "/" not in args.repo:
        parser.error("repo must be in owner/repo format")
    owner, name = args.repo.split("/", 1)
    collection = f"{owner}_{name}_{args.mode}"

    inspect(
        collection_name=collection,
        show_text=args.text,
        filter_type=args.filter_type,
        filter_file=args.filter_file,
        limit=args.limit,
    )
