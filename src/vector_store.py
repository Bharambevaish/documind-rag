"""
vector_store.py - all ChromaDB operations live here.

ChromaDB is a vector database: it stores each chunk's text, its metadata
(source file, page) and its embedding, and can quickly find the chunks whose
embeddings are closest to a question's embedding.

We use a *persistent* client, so the database is saved to `data/chroma/` and
survives app restarts - you don't have to re-upload documents every time.

No other module talks to ChromaDB directly. If we ever swap to another vector
DB (FAISS, Pinecone, pgvector...), only this file changes.
"""

from functools import lru_cache

import chromadb

from src.config import settings

COLLECTION_NAME = "documind"


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the collection once and reuse it."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Measure closeness with cosine distance (angle between vectors),
        # which is the standard choice for sentence embeddings.
        metadata={"hnsw:space": "cosine"},
        # We compute embeddings ourselves in embedder.py, so Chroma
        # shouldn't load its own embedding model.
        embedding_function=None,
    )


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Store chunks with their embeddings. `upsert` = insert, or overwrite same id."""
    if not chunks:
        return
    get_collection().upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Find the `top_k` chunks closest in meaning to the query, best first.

    Returns a list of {"text", "source", "page", "score"} where score is the
    cosine similarity (1.0 = identical meaning, ~0 = unrelated).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns lists-of-lists (one inner list per query); we sent one query.
    return [
        {
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            # Chroma gives cosine DISTANCE (0 = same). Similarity = 1 - distance.
            "score": round(1 - distance, 4),
        }
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def get_document_hash(source: str) -> str | None:
    """Return the stored content hash for a file name, or None if it isn't stored."""
    result = get_collection().get(where={"source": source}, limit=1, include=["metadatas"])
    return result["metadatas"][0]["doc_hash"] if result["ids"] else None


def find_source_by_hash(doc_hash: str) -> str | None:
    """Return the file name already stored with this exact content, if any."""
    result = get_collection().get(where={"doc_hash": doc_hash}, limit=1, include=["metadatas"])
    return result["metadatas"][0]["source"] if result["ids"] else None


def delete_document(source: str) -> int:
    """Delete every chunk of one file. Returns how many chunks were removed."""
    collection = get_collection()
    ids = collection.get(where={"source": source}, include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    """Summarise what's stored: one entry per file with its chunk and page counts."""
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for meta in metadatas:
        doc = docs.setdefault(meta["source"], {"source": meta["source"], "chunks": 0, "pages": set()})
        doc["chunks"] += 1
        doc["pages"].add(meta["page"])
    return [
        {"source": d["source"], "chunks": d["chunks"], "pages": len(d["pages"])}
        for d in sorted(docs.values(), key=lambda d: d["source"].lower())
    ]


def count_chunks() -> int:
    """Total number of chunks in the database."""
    return get_collection().count()


def clear_all() -> int:
    """Delete every chunk (used by the 'Clear all' button). Returns how many were removed."""
    collection = get_collection()
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)
