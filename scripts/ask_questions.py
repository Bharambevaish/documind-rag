"""
ask_questions.py - Phase 3 test: ask questions about the sample documents
from the command line.

Run the built-in test (3 questions in the docs + 1 that is NOT):
    python scripts/ask_questions.py

Or ask your own:
    python scripts/ask_questions.py "Can I carry forward unused leave?"

Expected: the first 3 answers cite the right file/page; the last one says
"I couldn't find this in the uploaded documents."
"""

import sys
from pathlib import Path

# Allow `from src...` imports when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import vector_store  # noqa: E402
from src.config import settings  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.rag_pipeline import answer_question  # noqa: E402

# (question, where the answer should come from)
TEST_QUESTIONS = [
    ("How many days of sick leave do employees get, and when is a medical certificate needed?",
     "company_leave_policy.pdf, p.2"),
    ("What is the difference between precision and recall?",
     "machine_learning_notes.pdf, p.4"),
    ("What are some ways to reduce overfitting?",
     "machine_learning_notes.pdf, p.3"),
    ("Who won the FIFA World Cup in 2022?",
     "NOT in the documents -> should say not found"),
]


def print_result(result: dict, expected: str | None = None) -> None:
    print(f"\nQ: {result['question']}")
    if expected:
        print(f"   (expected: {expected})")
    print(f"\nA: {result['answer']}\n")
    if result["sources"]:
        print("   Sources:")
        for s in result["sources"]:
            mark = "*" if s["cited"] else " "
            print(f"   {mark} {s['source']}, p.{s['page']}  (score {s['score']:.2f})")
        print("   (* = cited in the answer)")
    llm_note = "LLM called" if result["used_llm"] else "LLM skipped - best score below threshold"
    best = f"{result['best_score']:.2f}" if result["best_score"] is not None else "n/a"
    print(f"   Best score: {best} | {llm_note} | Time: {result['time_taken']}s")
    print("-" * 70)


def main() -> None:
    # The Windows console can't print some characters (e.g. curly quotes) by default.
    sys.stdout.reconfigure(encoding="utf-8")

    if vector_store.count_chunks() == 0:
        print("The vector store is empty. Run `python scripts/ingest_samples.py` first.")
        return

    print(f"Provider: {settings.llm_provider} | Model: {settings.model_name} | "
          f"Top-K: {settings.top_k} | Threshold: {settings.similarity_threshold}")
    print("-" * 70)

    questions = [(q, None) for q in sys.argv[1:]] or TEST_QUESTIONS
    for question, expected in questions:
        try:
            print_result(answer_question(question), expected)
        except LLMError as e:
            print(f"\nQ: {question}\n\nLLM ERROR: {e}")
            return


if __name__ == "__main__":
    main()
