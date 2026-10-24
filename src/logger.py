"""
logger.py - save every question and answer to logs/queries.jsonl.

JSONL = "JSON Lines": one JSON object per line. We only ever APPEND a line,
so writing is fast and a crash can't corrupt earlier entries. Reading it back
is simple too: one `json.loads` per line. Tools like pandas can load it directly.

The logs power the "Query Logs" tab and help answer questions like:
"How fast is the app?", "How often does it say not found?", "What do users ask?"
"""

import json
from datetime import datetime

from src.config import settings


def log_query(result: dict, top_k: int, error: str | None = None) -> None:
    """
    Append one entry for a question that was asked.

    Args:
        result: the dict returned by rag_pipeline.answer_question()
                (for failed questions, pass at least {"question": ...})
        top_k: how many chunks were retrieved for this question
        error: the error message if the question failed (e.g. LLM unreachable)
    """
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": result.get("question", ""),
        "answer": result.get("answer", ""),
        "found": result.get("found", False),
        "used_llm": result.get("used_llm", False),
        "best_score": result.get("best_score"),
        "time_taken": result.get("time_taken"),
        # Keep logs small: file + page + score only, not the full snippets.
        "sources": [
            {"source": s["source"], "page": s["page"], "score": s["score"]}
            for s in result.get("sources", [])
        ],
        "top_k": top_k,
        "provider": settings.llm_provider,
        "model": settings.model_name,
        "error": error,
    }
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_logs() -> list[dict]:
    """Return all log entries, newest first. Skips any damaged line instead of crashing."""
    if not settings.log_file.exists():
        return []
    entries = []
    with open(settings.log_file, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(entries))


def clear_logs() -> None:
    """Delete the log file (used by the 'Clear logs' button)."""
    settings.log_file.unlink(missing_ok=True)
