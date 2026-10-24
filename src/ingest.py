"""
ingest.py - the full ingestion pipeline for one file:

    load -> (duplicate check) -> chunk -> embed -> store

Both the Streamlit UI and the command-line scripts call `ingest_file`, so the
logic exists in exactly one place (and can be reused by the future Quiz Generator).

Duplicate handling uses a SHA-256 hash of the file's bytes - a fingerprint that
changes if even one byte of the file changes:
  * same content already stored (any file name)  -> "skipped"  (no wasted work)
  * same file name but the content changed        -> "replaced" (old chunks deleted first)
  * new file                                      -> "added"
"""

import hashlib
from pathlib import Path

from src import vector_store
from src.chunker import chunk_pages
from src.embedder import embed_texts
from src.loader import load_document


def ingest_file(file: str | Path | bytes, file_name: str | None = None) -> dict:
    """
    Ingest one document into the vector store.

    Args:
        file: path on disk, or raw bytes (e.g. from a Streamlit upload)
        file_name: required with bytes; defaults to the path's file name

    Returns:
        {"source": ..., "status": "added" | "replaced" | "skipped",
         "pages": int, "chunks": int, "message": str}
    """
    if isinstance(file, bytes):
        data = file
    else:
        file_name = file_name or Path(file).name
        data = Path(file).read_bytes()

    doc_hash = hashlib.sha256(data).hexdigest()

    # 1) Exact same content already stored? Then there's nothing to do.
    existing_source = vector_store.find_source_by_hash(doc_hash)
    if existing_source:
        return {
            "source": file_name, "status": "skipped", "pages": 0, "chunks": 0,
            "message": f"Already ingested (same content as '{existing_source}').",
        }

    # 2) Load and chunk first, so a broken file never deletes the old version.
    pages = load_document(data, file_name)
    chunks = chunk_pages(pages, doc_hash)

    # 3) Same name, new content -> remove the old chunks so no stale text remains.
    status = "added"
    if vector_store.get_document_hash(file_name):
        vector_store.delete_document(file_name)
        status = "replaced"

    # 4) Embed all chunks in one batch and store them.
    embeddings = embed_texts([c["text"] for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    return {
        "source": file_name, "status": status, "pages": len(pages), "chunks": len(chunks),
        "message": f"{status.capitalize()}: {len(pages)} pages -> {len(chunks)} chunks.",
    }
