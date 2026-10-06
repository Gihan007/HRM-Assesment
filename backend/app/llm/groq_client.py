"""Groq API client wrapper (primary LLM provider).

Raises `LLMConfigError` if GROQ_API_KEY isn't configured, and
`LLMRequestError` for any network/API failure, so `router.py` can catch
one common type regardless of provider.
"""

from __future__ import annotations

import groq

from app.core.config import get_settings
from app.llm import LLMConfigError, LLMRequestError

# Pinned explicitly so an ambient GROQ_BASE_URL env var (e.g. set globally
# on the host for some other tool) can't silently redirect/break requests.
# The Groq SDK appends its own "/openai/v1/..." path on top of base_url,
# so a base_url that already includes that suffix would double it.
_GROQ_BASE_URL = "https://api.groq.com"


def generate(messages: list[dict], *, temperature: float = 0.0) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMConfigError("GROQ_API_KEY is not configured.")

    client = groq.Groq(
        api_key=settings.groq_api_key,
        base_url=_GROQ_BASE_URL,
        timeout=settings.llm_timeout_seconds,
    )

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
        )
    except groq.GroqError as exc:
        raise LLMRequestError(f"Groq request failed: {exc}") from exc

    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice and choice.message else None
    if not content:
        raise LLMRequestError("Groq returned an empty response.")

    return content
