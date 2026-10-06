"""Unit tests for app.llm.router's Groq -> OpenRouter fallback behavior."""

from __future__ import annotations

import pytest

from app.llm import LLMError, LLMRequestError, router
from app.llm import groq_client, openrouter_client

MESSAGES = [{"role": "user", "content": "hello"}]


def test_router_returns_groq_result_when_groq_succeeds(monkeypatch):
    monkeypatch.setattr(groq_client, "generate", lambda messages, **kw: "from groq")
    monkeypatch.setattr(
        openrouter_client,
        "generate",
        lambda messages, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert router.generate_answer(MESSAGES) == "from groq"


def test_router_falls_back_to_openrouter_when_groq_fails(monkeypatch):
    def failing_groq(messages, **kw):
        raise LLMRequestError("groq down")

    monkeypatch.setattr(groq_client, "generate", failing_groq)
    monkeypatch.setattr(openrouter_client, "generate", lambda messages, **kw: "from openrouter")

    assert router.generate_answer(MESSAGES) == "from openrouter"


def test_router_raises_when_both_providers_fail(monkeypatch):
    def failing_groq(messages, **kw):
        raise LLMRequestError("groq down")

    def failing_openrouter(messages, **kw):
        raise LLMRequestError("openrouter down")

    monkeypatch.setattr(groq_client, "generate", failing_groq)
    monkeypatch.setattr(openrouter_client, "generate", failing_openrouter)

    with pytest.raises(LLMError):
        router.generate_answer(MESSAGES)
