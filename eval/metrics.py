"""Performance metrics computed from ``agent_logs.jsonl``.

Reads the structured JSONL log emitted by ``logger.log_step`` and produces
per-session and aggregate statistics (latency, step/tool counts, token
usage, and estimated Claude API spend).

Usage:
    python eval/metrics.py            # prints aggregate metrics as JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

LOG_FILE = Path("agent_logs.jsonl")

# claude-opus-4-7 pricing — USD per 1M tokens.
# Source: claude-api skill models table, cached 2026-04-15.
INPUT_PRICE_PER_M = 5.0
OUTPUT_PRICE_PER_M = 25.0


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000


def _load_all() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with LOG_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_session_metrics(session_id: str) -> dict:
    """For a single session: latency totals, step/tool counts, token usage, cost."""
    logs = [l for l in _load_all() if l["session_id"] == session_id]
    if not logs:
        return {}

    llm_latencies = [
        l["data"]["llm_latency_s"] for l in logs if "llm_latency_s" in l["data"]
    ]
    tool_latencies = [
        l["data"]["tool_latency_s"] for l in logs if "tool_latency_s" in l["data"]
    ]
    tool_calls = [l for l in logs if l["event"] == "tool_call"]
    final = next((l for l in logs if l["event"] == "final_answer"), None)

    input_tokens = final["data"].get("input_tokens", 0) if final else 0
    output_tokens = final["data"].get("output_tokens", 0) if final else 0

    return {
        "session_id": session_id,
        "total_steps": final["data"]["total_steps"] if final else None,
        "llm_calls": len(llm_latencies),
        "tool_calls": len(tool_calls),
        "total_llm_latency_s": round(sum(llm_latencies), 2),
        "total_tool_latency_s": round(sum(tool_latencies), 2),
        "avg_tool_latency_s": (
            round(mean(tool_latencies), 2) if tool_latencies else 0
        ),
        "total_latency_s": round(sum(llm_latencies) + sum(tool_latencies), 2),
        "answer_length": len(final["data"]["answer"]) if final else 0,
        "tools_used": sorted({l["data"]["tool"] for l in tool_calls}),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(
            _estimate_cost_usd(input_tokens, output_tokens), 6
        ),
    }


def compute_aggregate_metrics() -> dict:
    """Across all sessions in the log file."""
    all_logs = _load_all()
    if not all_logs:
        return {}

    sessions = {l["session_id"] for l in all_logs}
    per_session = [compute_session_metrics(s) for s in sessions]
    per_session = [s for s in per_session if s.get("total_latency_s")]

    if not per_session:
        return {}

    latencies = [s["total_latency_s"] for s in per_session]
    steps = [s["total_steps"] for s in per_session if s["total_steps"]]
    input_tokens_list = [s.get("input_tokens", 0) for s in per_session]
    output_tokens_list = [s.get("output_tokens", 0) for s in per_session]
    costs = [s.get("estimated_cost_usd", 0.0) for s in per_session]

    total_cost = sum(costs)
    return {
        "total_sessions": len(per_session),
        "avg_latency_s": round(mean(latencies), 2),
        "median_latency_s": round(median(latencies), 2),
        "p95_latency_s": (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
            if len(latencies) > 5
            else None
        ),
        "avg_steps": round(mean(steps), 1) if steps else 0,
        "total_input_tokens": sum(input_tokens_list),
        "total_output_tokens": sum(output_tokens_list),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query_usd": round(total_cost / len(per_session), 6),
    }


if __name__ == "__main__":
    print(json.dumps(compute_aggregate_metrics(), indent=2))
