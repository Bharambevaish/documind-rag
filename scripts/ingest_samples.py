"""
ingest_samples.py - Phase 2 test: ingest every file in data/sample_docs/
and print how many chunks were stored.

Run:  python scripts/ingest_samples.py
Run it twice: the second time every file should be "skipped" (duplicate check).
"""

import sys
import time
from pathlib import Path

# Allow `from src...` imports when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import vector_store  # noqa: E402
from src.config import settings  # noqa: E402
from src.ingest import ingest_file  # noqa: E402
from src.loader import SUPPORTED_EXTENSIONS  # noqa: E402


def main() -> None:
    files = sorted(p for p in settings.sample_docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        print(f"No documents in {settings.sample_docs_dir}. Run scripts/make_sample_docs.py first.")
        return

    print(f"Ingesting {len(files)} files (the first run downloads the embedding model)...\n")
    start = time.perf_counter()
    for path in files:
        result = ingest_file(path)
        print(f"  [{result['status']:>8}] {result['source']:<32} {result['message']}")
    print(f"\nDone in {time.perf_counter() - start:.1f}s\n")

    print("Documents in the vector store:")
    for doc in vector_store.list_documents():
        print(f"  {doc['source']:<32} {doc['pages']} pages, {doc['chunks']} chunks")
    print(f"\nTotal chunks stored: {vector_store.count_chunks()}")


if __name__ == "__main__":
    main()
