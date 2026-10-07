"""Stage 2: Query understanding.

Classifies whether retrieval is needed. Short greetings/pleasantries
("hi", "thanks", "bye") have no document-answerable content, and at
these lengths a bi-encoder's cosine similarity is unreliable (it can
land closer to a stylistically-conversational chunk than to any
topically relevant one), letting nonsense past the grounding gate. So
these are matched by exact phrase, not by embedding, and short-circuit
the pipeline before vector search ever runs.

Coreference resolution against short-term memory and explicit filter
extraction (filename, section, date) are not yet implemented.
"""

from __future__ import annotations

import re

_CHITCHAT_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "sup",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "hows it going",
    "whats up",
    "nice to meet you",
    "thanks",
    "thank you",
    "thanks a lot",
    "many thanks",
    "ok",
    "okay",
    "cool",
    "got it",
    "bye",
    "goodbye",
    "see you",
    "test",
    "ping",
}

_PUNCT_RE = re.compile(r"[^\w\s']")
_WHITESPACE_RE = re.compile(r"\s+")


def is_chitchat(question: str) -> bool:
    """Return True if `question` is a greeting/pleasantry with no
    document-answerable content, so retrieval should be skipped.

    Matches the whole (normalized) question against a fixed phrase
    list rather than doing substring/keyword matching, so a real
    question that happens to start with "hi" or "ok" is never
    misclassified.
    """
    normalized = _PUNCT_RE.sub("", question).strip().lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized in _CHITCHAT_PHRASES
