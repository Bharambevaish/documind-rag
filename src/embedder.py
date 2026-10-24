"""
embedder.py - turn text into vectors (lists of numbers) that capture meaning.

We use `all-MiniLM-L6-v2` from sentence-transformers:
  * free, runs on a normal laptop CPU, ~90 MB download (cached after the first run)
  * outputs a 384-number vector for any piece of text
  * texts with similar MEANING get vectors that point in similar directions,
    even if they use different words ("car" vs "automobile").

We normalise every vector to length 1. Then cosine similarity is just the dot
product, and scores are easy to compare (1.0 = same meaning, ~0 = unrelated).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes a few seconds)."""
    return SentenceTransformer(settings.embedding_model, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts at once (batching is much faster than one by one)."""
    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single question. Must use the SAME model as the documents."""
    return embed_texts([text])[0]
