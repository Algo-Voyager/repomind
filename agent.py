"""Core agent orchestrator — ReAct-style tool-use loop against Claude.

One agent, one loop. Given a user question and a Chroma collection name,
rewrites the query for semantic search, then iterates:

    LLM ──▶ tool_use? ──▶ run_tool ──▶ tool_result ──▶ LLM ...

...until the model returns ``stop_reason == "end_turn"`` or we hit
``max_steps``. Every LLM call, tool call, and tool result is logged to
``agent_logs.jsonl`` via :func:`logger.log_step`.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import anthropic
from dotenv import load_dotenv

from logger import log_step
from prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, run_tool

load_dotenv()

MODEL = "claude-opus-4-7"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazy client init so imports don't fail when the key is absent."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def query_rewrite(user_query: str) -> str:
    """Rewrite a user question into a compact semantic-search query."""
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": QUERY_REWRITE_PROMPT.format(query=user_query),
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _extract_text(blocks: list[Any]) -> str:
    return "".join(b.text for b in blocks if getattr(b, "type", None) == "text")


def run_agent(
    user_query: str,
    collection_name: str,
    max_steps: int = 6,
) -> dict[str, Any]:
    """Drive the agent loop until the model finishes or ``max_steps`` is hit."""
    client = _get_client()
    session_id = str(uuid.uuid4())[:8]

    rewritten = query_rewrite(user_query)
    log_step(session_id, 0, "query_rewrite", {
        "original": user_query,
        "rewritten": rewritten,
    })

    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"Original question: {user_query}\n"
            f"Optimized search query: {rewritten}"
        ),
    }]

    for step in range(1, max_steps + 1):
        t0 = time.time()
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        llm_latency = round(time.time() - t0, 2)

        if response.stop_reason == "end_turn":
            answer = _extract_text(response.content)
            log_step(session_id, step, "final_answer", {
                "answer": answer,
                "llm_latency_s": llm_latency,
                "total_steps": step,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })
            return {
                "session_id": session_id,
                "answer": answer,
                "steps": step,
                "messages": messages,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        if response.stop_reason == "refusal":
            log_step(session_id, step, "refusal", {"llm_latency_s": llm_latency})
            return {
                "session_id": session_id,
                "answer": "The model declined to answer for safety reasons.",
                "steps": step,
                "messages": messages,
            }

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            # max_tokens, pause_turn, stop_sequence, or anything else —
            # no tool to run and no end_turn, so surface what we have.
            answer = _extract_text(response.content)
            log_step(session_id, step, "unexpected_stop", {
                "stop_reason": response.stop_reason,
                "llm_latency_s": llm_latency,
            })
            return {
                "session_id": session_id,
                "answer": answer or f"[stopped: {response.stop_reason}]",
                "steps": step,
                "messages": messages,
            }

        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            args = dict(tool_use.input)
            log_step(session_id, step, "tool_call", {
                "tool": tool_use.name,
                "args": args,
                "llm_latency_s": llm_latency,
            })

            t1 = time.time()
            result = run_tool(tool_use.name, args, collection_name)
            tool_latency = round(time.time() - t1, 2)

            result_str = str(result)
            log_step(session_id, step, "tool_result", {
                "tool": tool_use.name,
                "result_preview": result_str[:200],
                "result_chars": len(result_str),
                "tool_latency_s": tool_latency,
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    log_step(session_id, max_steps, "max_steps_reached", {})
    return {
        "session_id": session_id,
        "answer": "I couldn't find a complete answer after max steps.",
        "steps": max_steps,
        "messages": messages,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <collection_name> <query...>", file=sys.stderr)
        sys.exit(2)
    collection = sys.argv[1]
    query = " ".join(sys.argv[2:])
    result = run_agent(query, collection)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result["answer"])
    print(f"\nSteps: {result['steps']}")
    if "usage" in result:
        print(
            f"Tokens: {result['usage']['input_tokens']} in / "
            f"{result['usage']['output_tokens']} out"
        )
