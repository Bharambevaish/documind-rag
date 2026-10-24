"""
chunker.py - split pages into small, overlapping chunks.

Why chunk at all?
  * An embedding of a whole 30-page PDF would be a blurry average of everything.
    Small chunks (~800 chars, about a paragraph or two) each capture one idea,
    so search can find the exact passage that answers a question.
  * The LLM only gets the top few chunks, which keeps the prompt short and cheap.

Why overlap (~150 chars)?
  * A sentence that falls on the boundary between two chunks would otherwise be
    cut in half. Overlap repeats the end of one chunk at the start of the next,
    so no idea is lost at the edges.

We split each page separately, so every chunk belongs to exactly ONE page and
its citation ("p.3") is always accurate.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

# "Recursive" = try to split on paragraphs first, then lines, then sentences,
# then words, and only cut mid-word as a last resort. This keeps chunks readable.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[dict], doc_hash: str) -> list[dict]:
    """
    Split pages into chunks, keeping the metadata of the page each chunk came from.

    Args:
        pages: output of loader.load_document()
        doc_hash: fingerprint of the file's contents (see ingest.py), used to
                  build unique, repeatable chunk ids.

    Returns a list of:
        {"id": "3f2a9c1b7d4e-p2-c0",
         "text": "...",
         "metadata": {"source": "notes.pdf", "page": 2, "chunk_index": 0, "doc_hash": "..."}}
    """
    chunks = []
    for page in pages:
        for index, text in enumerate(_splitter.split_text(page["text"])):
            chunks.append({
                # Same file -> same ids every time. Short hash + page + position
                # is unique within the database and easy to read when debugging.
                "id": f"{doc_hash[:12]}-p{page['page']}-c{index}",
                "text": text,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": index,
                    "doc_hash": doc_hash,
                },
            })
    return chunks
