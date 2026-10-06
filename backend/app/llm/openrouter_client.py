"""OpenRouter API client wrapper (fallback LLM provider).

OpenRouter exposes an OpenAI-compatible REST endpoint; there's no
official SDK, so this calls it directly over httpx. Same error contract
as groq_client: `LLMConfigError` for a missing key, `LLMRequestError`
for any network/API failure.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.llm import LLMConfigError, LLMRequestError

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate(messages: list[dict], *, temperature: float = 0.0) -> str:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMConfigError("OPENROUTER_API_KEY is not configured.")

    try:
        response = httpx.post(
            _OPENROUTER_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.openrouter_model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"OpenRouter request failed: {exc}") from exc

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMRequestError(f"OpenRouter returned an unexpected response shape: {data!r}") from exc

    if not content:
        raise LLMRequestError("OpenRouter returned an empty response.")

    return content
