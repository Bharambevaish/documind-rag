"""
llm.py - talk to the Large Language Model through ONE function: `generate()`.

Two providers are supported, chosen with LLM_PROVIDER in .env:
  * groq   - cloud API, free tier, very fast (default)
  * ollama - runs a model on your own machine, no internet or API key needed

The rest of the app only calls `generate()` and never knows which provider is
used. Adding a new provider (OpenAI, Gemini, ...) means adding one function here.

Every failure is turned into an `LLMError` with a friendly message, so the UI
can show "API key missing" instead of a long stack trace.
"""

import re

import requests

from src.config import settings

# Low temperature = less "creative", more factual and repeatable answers.
# That's what we want when answering strictly from documents.
TEMPERATURE = 0.1
MAX_TOKENS = 512        # upper limit on answer length
TIMEOUT_SECONDS = 60


class LLMError(Exception):
    """Raised with a human-readable message when the LLM can't be reached or used."""


def _generate_groq(messages: list[dict]) -> str:
    # Imported here so people using only Ollama don't need the groq package to work.
    import groq

    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is missing. Add it to your .env file "
                       "(free key: https://console.groq.com/keys).")

    client = groq.Groq(api_key=settings.groq_api_key, timeout=TIMEOUT_SECONDS)
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except groq.AuthenticationError:
        raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.") from None
    except groq.RateLimitError:
        raise LLMError("Groq rate limit reached (free tier). Wait a minute and try again.") from None
    except groq.NotFoundError:
        raise LLMError(f"Groq model '{settings.model_name}' not found. Check MODEL_NAME in .env.") from None
    except groq.APIConnectionError:
        raise LLMError("Could not connect to Groq. Check your internet connection.") from None
    except groq.APIError as e:
        raise LLMError(f"Groq API error: {e}") from None

    return response.choices[0].message.content.strip()


def _generate_ollama(messages: list[dict]) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "stream": False,  # wait for the full answer instead of token-by-token
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.ConnectionError:
        raise LLMError(f"Could not reach Ollama at {settings.ollama_base_url}. "
                       "Is it installed and running (`ollama serve`)?") from None
    except requests.Timeout:
        raise LLMError("Ollama took too long to answer. Try a smaller model.") from None

    if response.status_code == 404:
        raise LLMError(f"Ollama model '{settings.model_name}' not found. "
                       f"Run `ollama pull {settings.model_name}` first.")
    if not response.ok:
        raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")

    return response.json()["message"]["content"].strip()


def generate(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to the configured LLM and return its text answer.

    Args:
        prompt: the user message (for RAG: the context chunks + the question)
        system_prompt: standing instructions on how the model must behave
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if settings.llm_provider == "ollama":
        answer = _generate_ollama(messages)
    else:
        answer = _generate_groq(messages)

    # Some "reasoning" models (Qwen3, DeepSeek-R1...) print their private
    # thinking inside <think>...</think> before the answer. Users only want the answer.
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
