"""
config.py - loads all settings in ONE place.

Every other module does `from src.config import settings` instead of reading
environment variables itself. That way:
  * secrets (the API key) live only in the git-ignored `.env` file,
  * changing a setting never means hunting through the code,
  * bad values are caught early with a clear error message.

Run `python -m src.config` to print the loaded settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `src/`. Building paths from here means
# the app works no matter which folder you start it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read key=value pairs from `.env` into environment variables.
# Real environment variables win over .env (override=False), which is handy
# on servers where you set secrets in the hosting dashboard instead.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Default model for each provider, used when MODEL_NAME is left empty.
DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "ollama": "llama3",
}


@dataclass(frozen=True)  # frozen = settings can't be changed by accident at runtime
class Settings:
    # --- LLM ---
    llm_provider: str
    groq_api_key: str
    model_name: str
    ollama_base_url: str

    # --- Retrieval ---
    top_k: int

    # --- Fixed project settings (from PLAN.md, not meant to change per user) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Paths ---
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    log_file: Path = PROJECT_ROOT / "logs" / "queries.jsonl"

    def masked_api_key(self) -> str:
        """Show only the last 4 characters so the key is never printed in full."""
        if not self.groq_api_key:
            return "(not set)"
        return "*" * 8 + self.groq_api_key[-4:]


def _read_int(name: str, default: int) -> int:
    """Read an integer env variable, with a clear error if it isn't a number."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got '{raw}'") from None


def load_settings() -> Settings:
    """Read the environment, apply defaults, validate, and return a Settings object."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be 'groq' or 'ollama', got '{provider}'")

    top_k = _read_int("TOP_K", 4)
    if top_k < 1:
        raise ValueError(f"TOP_K must be at least 1, got {top_k}")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    # Treat the placeholder from .env.example as "not set".
    if api_key == "your_groq_api_key_here":
        api_key = ""

    return Settings(
        llm_provider=provider,
        groq_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "").strip() or DEFAULT_MODELS[provider],
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        top_k=top_k,
    )


# Load once when the module is first imported; everyone shares this object.
settings = load_settings()


if __name__ == "__main__":
    # Phase 1 test: print the loaded config (with the API key masked).
    print("DocuMind configuration")
    print("-" * 40)
    print(f"LLM provider     : {settings.llm_provider}")
    print(f"Model name       : {settings.model_name}")
    print(f"Groq API key     : {settings.masked_api_key()}")
    print(f"Ollama URL       : {settings.ollama_base_url}")
    print(f"Top-K            : {settings.top_k}")
    print(f"Embedding model  : {settings.embedding_model}")
    print(f"Chunk size       : {settings.chunk_size} chars (overlap {settings.chunk_overlap})")
    print(f"Chroma dir       : {settings.chroma_dir}")
    print(f"Log file         : {settings.log_file}")

    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("\nWARNING: GROQ_API_KEY is not set. Add it to your .env file before Phase 3.")
