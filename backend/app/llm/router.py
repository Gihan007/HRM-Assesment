"""Tries Groq first, falls back to OpenRouter on error/timeout/rate-limit.

Either provider failing in isolation (missing key, rate limit, network
error, timeout) should not break generation if the other one works.
Only if both fail does this raise, so the caller has one place to
handle "the LLM is unavailable" rather than per-provider error cases.
"""

from __future__ import annotations

import logging

from app.llm import LLMError
from app.llm import groq_client, openrouter_client

logger = logging.getLogger(__name__)


def generate_answer(messages: list[dict], *, temperature: float = 0.0) -> str:
    try:
        return groq_client.generate(messages, temperature=temperature)
    except LLMError as groq_exc:
        logger.warning("Groq generation failed, falling back to OpenRouter: %s", groq_exc)

    try:
        return openrouter_client.generate(messages, temperature=temperature)
    except LLMError as openrouter_exc:
        raise LLMError(
            "Both Groq and OpenRouter failed to generate a response. "
            f"OpenRouter error: {openrouter_exc}"
        ) from openrouter_exc
