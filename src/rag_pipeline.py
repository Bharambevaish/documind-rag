"""
rag_pipeline.py - Retrieval-Augmented Generation: answer a question from the documents.

    question
      -> embed the question                 (embedder.py)
      -> retrieve the top-K closest chunks  (vector_store.py)
      -> guard: best score too low?  -> answer "not found" WITHOUT calling the LLM
      -> build a prompt: rules + numbered context chunks + question
      -> generate the answer                (llm.py)
      -> return answer + sources (file, page, snippet, score) + time taken

Two layers stop the model from making things up (hallucinating):
  1. Similarity threshold - unrelated questions never reach the LLM at all.
  2. Strict prompt - the LLM must answer ONLY from the given context, cite it,
     and say NOT_FOUND_MESSAGE when the context doesn't contain the answer.
"""

import time

from src import vector_store
from src.config import settings
from src.embedder import embed_query
from src.llm import generate

NOT_FOUND_MESSAGE = "I couldn't find this in the uploaded documents."
SNIPPET_LENGTH = 300  # characters of each source chunk to show the user

SYSTEM_PROMPT = f"""You are DocuMind, an assistant that answers questions about the user's documents.

Rules:
1. Answer ONLY using the information in the CONTEXT below. Do not use outside knowledge.
2. After each fact, cite its source exactly as it appears in the context header, e.g. [notes.pdf, p.3].
3. If the context does not contain the answer, reply with exactly this sentence and nothing else:
{NOT_FOUND_MESSAGE}
4. Be concise and clear. Use short bullet points when listing several items."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Put the retrieved chunks and the question together into one user message."""
    context_blocks = [
        f"[{chunk['source']}, p.{chunk['page']}]\n{chunk['text']}"
        for chunk in chunks
    ]
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


def _is_not_found(answer: str) -> bool:
    """True if the LLM said the answer isn't in the documents (ignores curly quotes / case)."""
    normalised = answer.replace("’", "'").lower()
    return "couldn't find this in the uploaded documents" in normalised


def _format_sources(chunks: list[dict], answer: str) -> list[dict]:
    """Shape retrieved chunks for display, flagging the ones the answer actually cites."""
    return [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "snippet": chunk["text"][:SNIPPET_LENGTH] + ("..." if len(chunk["text"]) > SNIPPET_LENGTH else ""),
            "score": chunk["score"],
            "cited": f"[{chunk['source']}, p.{chunk['page']}]" in answer,
        }
        for chunk in chunks
    ]


def answer_question(question: str, top_k: int | None = None) -> dict:
    """
    Answer a question using the ingested documents.

    Returns:
        {"question", "answer", "found": bool, "used_llm": bool,
         "sources": [{"source", "page", "snippet", "score", "cited"}],
         "best_score": float | None, "time_taken": seconds}

    Raises:
        ValueError: empty question
        llm.LLMError: the LLM couldn't be reached (missing key, no internet, ...)
    """
    question = question.strip()
    if not question:
        raise ValueError("Please type a question.")

    start = time.perf_counter()
    top_k = top_k or settings.top_k

    # 1) RETRIEVE: embed the question and find the closest chunks.
    chunks = vector_store.search(embed_query(question), top_k)
    best_score = chunks[0]["score"] if chunks else None

    def result(answer: str, found: bool, used_llm: bool, sources: list[dict]) -> dict:
        return {
            "question": question,
            "answer": answer,
            "found": found,
            "used_llm": used_llm,
            "sources": sources,
            "best_score": best_score,
            "time_taken": round(time.perf_counter() - start, 2),
        }

    # 2) GUARDS: nothing uploaded, or nothing relevant -> no LLM call (saves time + money).
    if not chunks:
        return result("No documents have been uploaded yet. Please upload a file first.", False, False, [])
    if best_score < settings.similarity_threshold:
        return result(NOT_FOUND_MESSAGE, False, False, [])

    # Drop weak chunks so they can't distract the LLM.
    relevant = [c for c in chunks if c["score"] >= settings.similarity_threshold]

    # 3) GENERATE: ask the LLM to answer from the relevant chunks only.
    answer = generate(build_prompt(question, relevant), SYSTEM_PROMPT)

    # If the LLM itself says "not found", showing sources would be misleading.
    if _is_not_found(answer):
        return result(NOT_FOUND_MESSAGE, False, True, [])

    return result(answer, True, True, _format_sources(relevant, answer))
