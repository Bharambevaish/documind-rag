"""
evaluate.py - measure how well DocuMind works, with numbers instead of "it seems fine".

For every question in eval/test_questions.json we run:
  1. RETRIEVAL ONLY - is the expected (file, page) among the top-K chunks?
  2. THE FULL PIPELINE - answer_question(): is the answer correct and cited,
     or correctly "not found"? How long did it take?

Metrics (printed and saved to eval/results.md):
  * Retrieval hit rate @K - % of answerable questions whose expected page was retrieved.
                            If retrieval misses, the LLM can't possibly answer correctly.
  * MRR (Mean Reciprocal Rank) - average of 1/rank of the first correct chunk
                            (1.0 = always ranked first, 0.5 = usually second, 0 = never found).
  * Citation accuracy     - % of answerable questions answered AND citing the expected page.
  * Correct "not found"   - % of out-of-docs questions where the app refused instead of guessing.
  * Avg response time     - end-to-end seconds per question (retrieval + LLM).

Run (after `python scripts/ingest_samples.py`):
    python eval/evaluate.py
    python eval/evaluate.py --top-k 2
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow `from src...` imports when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import vector_store  # noqa: E402
from src.config import settings  # noqa: E402
from src.embedder import embed_query  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.rag_pipeline import answer_question  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "test_questions.json"
RESULTS_FILE = EVAL_DIR / "results.md"


def evaluate_question(item: dict, top_k: int) -> dict:
    """Run retrieval and the full pipeline for one test question."""
    question = item["question"]
    answerable = item["expected_source"] is not None

    # 1) Retrieval only: where (if anywhere) does the expected page rank?
    rank = None
    if answerable:
        chunks = vector_store.search(embed_query(question), top_k)
        for position, chunk in enumerate(chunks, start=1):
            if chunk["source"] == item["expected_source"] and chunk["page"] == item["expected_page"]:
                rank = position
                break

    # 2) Full pipeline, exactly as the app runs it.
    try:
        result = answer_question(question, top_k=top_k)
    except LLMError as e:
        return {**item, "answerable": answerable, "rank": rank, "error": str(e),
                "passed": False, "time_taken": None, "used_llm": True, "answer": ""}

    if answerable:
        citation = f"[{item['expected_source']}, p.{item['expected_page']}]"
        passed = result["found"] and citation in result["answer"]
    else:
        passed = not result["found"]

    return {**item, "answerable": answerable, "rank": rank, "error": None, "passed": passed,
            "time_taken": result["time_taken"], "used_llm": result["used_llm"],
            "best_score": result["best_score"], "answer": result["answer"]}


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}% ({part}/{whole})" if whole else "n/a"


def summarise(rows: list[dict], top_k: int) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    times = [r["time_taken"] for r in rows if r["time_taken"] is not None]
    return {
        f"Retrieval hit rate @{top_k}": percent(sum(r["rank"] is not None for r in answerable), len(answerable)),
        "MRR (mean reciprocal rank)": f"{sum(1 / r['rank'] for r in answerable if r['rank']) / len(answerable):.2f}"
                                      if answerable else "n/a",
        "Citation accuracy": percent(sum(r["passed"] for r in answerable), len(answerable)),
        'Correct "not found"': percent(sum(r["passed"] for r in unanswerable), len(unanswerable)),
        "Avg response time": f"{sum(times) / len(times):.2f}s" if times else "n/a",
        "Slowest response": f"{max(times):.2f}s" if times else "n/a",
        "LLM calls skipped (threshold)": f"{sum(not r['used_llm'] for r in rows)}/{len(rows)}",
        "Errors": str(sum(r["error"] is not None for r in rows)),
    }


def write_markdown(rows: list[dict], summary: dict, top_k: int) -> None:
    lines = [
        "# DocuMind - Evaluation Results",
        "",
        f"- **Date:** {datetime.now():%Y-%m-%d %H:%M}",
        f"- **LLM:** {settings.llm_provider} / `{settings.model_name}`",
        f"- **Embedding model:** `{settings.embedding_model}`",
        f"- **Top-K:** {top_k} | **Similarity threshold:** {settings.similarity_threshold} | "
        f"**Chunk size / overlap:** {settings.chunk_size} / {settings.chunk_overlap}",
        f"- **Questions:** {len(rows)} ({sum(r['answerable'] for r in rows)} answerable, "
        f"{sum(not r['answerable'] for r in rows)} not in the documents)",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in summary.items()],
        "",
        "## Per question",
        "",
        "| # | Question | Expected | Retrieved at rank | Result | Time (s) | LLM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        expected = f"{r['expected_source']} p.{r['expected_page']}" if r["answerable"] else "not found"
        rank = (str(r["rank"]) if r["rank"] else "missed") if r["answerable"] else "-"
        status = "⚠️ error" if r["error"] else ("✅" if r["passed"] else "❌")
        time_taken = f"{r['time_taken']:.2f}" if r["time_taken"] is not None else "-"
        llm = "called" if r["used_llm"] else "skipped"
        lines.append(f"| {r['id']} | {r['question']} | {expected} | {rank} | {status} | {time_taken} | {llm} |")

    failures = [r for r in rows if not r["passed"]]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None - every question passed.")
    for r in failures:
        reason = r["error"] or f"Answer: {r['answer'][:300]}"
        lines.append(f"- **Q{r['id']}** {r['question']} - {reason}")

    lines += ["", "_Generated by `python eval/evaluate.py`._", ""]
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console + emoji / curly quotes
    parser = argparse.ArgumentParser(description="Evaluate DocuMind retrieval and answers.")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="chunks to retrieve")
    top_k = parser.parse_args().top_k

    if vector_store.count_chunks() == 0:
        print("The vector store is empty. Run `python scripts/ingest_samples.py` first.")
        return

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    print(f"Evaluating {len(questions)} questions | {settings.llm_provider}/{settings.model_name} | Top-K {top_k}\n")

    rows = []
    for item in questions:
        row = evaluate_question(item, top_k)
        rows.append(row)
        status = "ERROR" if row["error"] else ("PASS" if row["passed"] else "FAIL")
        time_taken = f"{row['time_taken']:.2f}s" if row["time_taken"] is not None else "-"
        print(f"  [{status:>5}] Q{row['id']:<2} {time_taken:>6}  {row['question']}")

    summary = summarise(rows, top_k)
    print("\nSummary")
    for name, value in summary.items():
        print(f"  {name:<32} {value}")

    write_markdown(rows, summary, top_k)
    print(f"\nSaved to {RESULTS_FILE.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
