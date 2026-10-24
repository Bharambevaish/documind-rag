"""
loader.py - turn a file (PDF / TXT / MD) into a list of pages.

Each page is a small dict:
    {"text": "...", "source": "notes.pdf", "page": 3}

Keeping `source` and `page` from the very start is what lets us show
"[notes.pdf, p.3]" next to every answer later on.

Files can come from disk (a path) or from a Streamlit upload (raw bytes),
so `load_document` accepts both.
"""

import io
import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _clean(text: str) -> str:
    """Tidy up extracted text: collapse runs of spaces/tabs and of blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _load_pdf(data: bytes, file_name: str) -> list[dict]:
    """Read a PDF page by page. Page numbers start at 1, like in a PDF viewer."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        # Skip empty pages (e.g. blank pages or scanned images with no text layer).
        if text:
            pages.append({"text": text, "source": file_name, "page": page_number})
    return pages


def _load_text(data: bytes, file_name: str) -> list[dict]:
    """TXT and MD files have no pages, so the whole file is treated as page 1."""
    # errors="replace" means a stray bad byte won't crash the whole upload.
    text = _clean(data.decode("utf-8", errors="replace"))
    return [{"text": text, "source": file_name, "page": 1}] if text else []


def load_document(file: str | Path | bytes, file_name: str | None = None) -> list[dict]:
    """
    Load one document and return its pages.

    Args:
        file: a path on disk, OR the raw bytes of an uploaded file.
        file_name: required when passing bytes (used as the `source` label).

    Raises:
        ValueError: if the file type isn't supported or no text could be extracted.
    """
    if isinstance(file, bytes):
        if not file_name:
            raise ValueError("file_name is required when loading from bytes")
        data = file
    else:
        path = Path(file)
        file_name = file_name or path.name
        data = path.read_bytes()

    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{extension}'. Use PDF, TXT or MD.")

    pages = _load_pdf(data, file_name) if extension == ".pdf" else _load_text(data, file_name)

    if not pages:
        raise ValueError(f"No text found in '{file_name}'. (Scanned image PDFs need OCR, which isn't supported.)")
    return pages
