"""Unit tests for app.llm.groq_client and app.llm.openrouter_client.
No real network calls are made: the Groq SDK client and httpx.post are
both replaced with fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import groq
import httpx
import pytest

from app.llm import groq_client, openrouter_client
from app.llm import LLMConfigError, LLMRequestError

MESSAGES = [{"role": "user", "content": "hello"}]


def _fake_settings(**overrides):
    base = dict(
        groq_api_key="fake-groq-key",
        groq_model="test-model",
        openrouter_api_key="fake-openrouter-key",
        openrouter_model="test-model",
        llm_timeout_seconds=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- groq_client ---------------------------------------------------------


def test_groq_generate_raises_config_error_when_key_missing(monkeypatch):
    monkeypatch.setattr(groq_client, "get_settings", lambda: _fake_settings(groq_api_key=""))

    with pytest.raises(LLMConfigError):
        groq_client.generate(MESSAGES)


def test_groq_generate_returns_content_on_success(monkeypatch):
    monkeypatch.setattr(groq_client, "get_settings", lambda: _fake_settings())

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["messages"] == MESSAGES
            message = SimpleNamespace(content="Paris is the capital of France. [1]")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(groq_client.groq, "Groq", FakeGroq)

    result = groq_client.generate(MESSAGES)
    assert result == "Paris is the capital of France. [1]"


def test_groq_generate_wraps_api_errors(monkeypatch):
    monkeypatch.setattr(groq_client, "get_settings", lambda: _fake_settings())

    class FakeCompletions:
        def create(self, **kwargs):
            raise groq.GroqError("rate limited")

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(groq_client.groq, "Groq", FakeGroq)

    with pytest.raises(LLMRequestError):
        groq_client.generate(MESSAGES)


def test_groq_generate_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(groq_client, "get_settings", lambda: _fake_settings())

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[])

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(groq_client.groq, "Groq", FakeGroq)

    with pytest.raises(LLMRequestError):
        groq_client.generate(MESSAGES)


# --- openrouter_client -----------------------------------------------------


def test_openrouter_generate_raises_config_error_when_key_missing(monkeypatch):
    monkeypatch.setattr(
        openrouter_client, "get_settings", lambda: _fake_settings(openrouter_api_key="")
    )

    with pytest.raises(LLMConfigError):
        openrouter_client.generate(MESSAGES)


def test_openrouter_generate_returns_content_on_success(monkeypatch):
    monkeypatch.setattr(openrouter_client, "get_settings", lambda: _fake_settings())

    def fake_post(url, *, headers, json, timeout):
        assert "Bearer fake-openrouter-key" == headers["Authorization"]
        assert json["model"] == "test-model"
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Paris. [1]"}}]},
            request=request,
        )

    monkeypatch.setattr(openrouter_client.httpx, "post", fake_post)

    result = openrouter_client.generate(MESSAGES)
    assert result == "Paris. [1]"


def test_openrouter_generate_wraps_http_errors(monkeypatch):
    monkeypatch.setattr(openrouter_client, "get_settings", lambda: _fake_settings())

    def fake_post(url, *, headers, json, timeout):
        request = httpx.Request("POST", url)
        return httpx.Response(429, json={"error": "rate limited"}, request=request)

    monkeypatch.setattr(openrouter_client.httpx, "post", fake_post)

    with pytest.raises(LLMRequestError):
        openrouter_client.generate(MESSAGES)


def test_openrouter_generate_wraps_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(openrouter_client, "get_settings", lambda: _fake_settings())

    def fake_post(url, *, headers, json, timeout):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"unexpected": "shape"}, request=request)

    monkeypatch.setattr(openrouter_client.httpx, "post", fake_post)

    with pytest.raises(LLMRequestError):
        openrouter_client.generate(MESSAGES)
