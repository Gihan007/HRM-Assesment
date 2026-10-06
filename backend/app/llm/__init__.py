"""LLM clients: Groq (primary) and OpenRouter (fallback), with routing logic.

Shared exception types live here so both provider clients and `router`
can raise/catch a common hierarchy without importing each other.
"""


class LLMError(Exception):
    """Base class for all LLM-provider errors."""


class LLMConfigError(LLMError):
    """Raised when a required API key/config is missing."""


class LLMRequestError(LLMError):
    """Raised when the provider API call fails: network error, timeout,
    auth failure, rate limit, or an unexpected response shape.
    """
