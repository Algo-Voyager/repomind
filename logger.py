"""Structured JSONL logging for agent runs.

Each call to ``log_step`` appends one JSON object per line to
``agent_logs.jsonl`` (override with the ``AGENT_LOG_FILE`` env var).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FILE = Path(os.getenv("AGENT_LOG_FILE", "agent_logs.jsonl"))


def log_step(
    session_id: str,
    step: int,
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Append one structured event to the log file."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "step": step,
        "event": event_type,
        "data": data,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
