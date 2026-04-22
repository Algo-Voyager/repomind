"""Correctness test suite — runs pre-defined queries against the agent
and checks that answers contain / avoid expected keywords.

Usage:
    python eval/test_queries.py <collection_name>

Writes ``eval_results.jsonl`` (gitignored) with one JSON object per query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import path: ``python eval/test_queries.py`` runs with CWD at the project
# root (see Gate 4 in the README), so ``agent`` resolves directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import run_agent  # noqa: E402

TEST_QUERIES = [
    {
        "id": "q1",
        "question": "What is the main purpose of this repository?",
        "must_contain_any": ["overview", "purpose", "what"],
        "must_not_contain": ["I don't know", "cannot find", "unable to"],
    },
    {
        "id": "q2",
        "question": "Show me the main entry point of the application",
        "must_contain_any": ["main", "entry", "index", "app"],
        "must_not_contain": ["I don't know"],
    },
    {
        "id": "q3",
        "question": "What are the main dependencies used?",
        "must_contain_any": ["dependencies", "packages", "require", "import"],
        "must_not_contain": ["I don't know"],
    },
    {
        "id": "q4",
        "question": "Are there any test files? What do they test?",
        "must_contain_any": ["test", "spec", "no tests"],
        "must_not_contain": [],
    },
    {
        "id": "q5",
        "question": "Show me a specific function and explain what it does",
        "must_contain_any": ["function", "def ", "const ", "```"],
        "must_not_contain": ["I don't know"],
    },
]


def run_tests(collection_name: str) -> list[dict]:
    results: list[dict] = []
    for test in TEST_QUERIES:
        print(f"\n{'=' * 60}\nRunning: {test['id']} — {test['question']}")
        try:
            result = run_agent(test["question"], collection_name)
            answer = result["answer"].lower()

            passes_must = (
                not test["must_contain_any"]
                or any(kw.lower() in answer for kw in test["must_contain_any"])
            )
            passes_not = not any(
                kw.lower() in answer for kw in test["must_not_contain"]
            )

            passed = passes_must and passes_not

            results.append({
                "id": test["id"],
                "question": test["question"],
                "passed": passed,
                "steps": result["steps"],
                "answer_preview": result["answer"][:200],
                "session_id": result["session_id"],
            })
            print(f"{'✅ PASS' if passed else '❌ FAIL'} | steps={result['steps']}")
        except Exception as e:
            results.append({
                "id": test["id"],
                "passed": False,
                "error": str(e),
                "question": test["question"],
            })
            print(f"❌ ERROR: {e}")

    passed_count = sum(1 for r in results if r.get("passed"))
    pass_rate = passed_count / len(results)
    print(
        f"\n{'=' * 60}\n"
        f"PASS RATE: {pass_rate * 100:.0f}% ({passed_count}/{len(results)})"
    )

    Path("eval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n",
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval/test_queries.py <collection_name>", file=sys.stderr)
        sys.exit(2)
    run_tests(sys.argv[1])
