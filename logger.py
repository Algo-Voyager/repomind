"""Structured JSONL logging for agent runs.

Each call to ``log_step`` appends one JSON object per line to
``agent_logs.jsonl`` (override with the ``AGENT_LOG_FILE`` env var).
``get_recent_logs`` and ``get_session_logs`` are thin readers used by
the Streamlit UI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

LOG_FILE = Path(os.getenv("AGENT_LOG_FILE", "agent_logs.jsonl"))


def log_step(
    session_id: str,
    step: int,
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Append one structured event to the log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "step": step,
        "event": event_type,
        "data": data,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _iter_entries() -> Iterator[dict]:
    if not LOG_FILE.exists():
        return
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_recent_logs(n: int = 50) -> list[dict]:
    """Return the last ``n`` log entries in chronological order."""
    return list(_iter_entries())[-n:]


def get_session_logs(session_id: str) -> list[dict]:
    """Return every entry belonging to ``session_id`` in the order written."""
    return [e for e in _iter_entries() if e["session_id"] == session_id]
