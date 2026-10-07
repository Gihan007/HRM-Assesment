"""Stage 3: Query expansion / rewriting.

Rewrites the user's raw message into a clean, keyword-focused search
query before it goes into retrieval. This matters because the
downstream retrieval+rerank scores are sensitive to how the query is
phrased: conversational wrapping ("can u tell me about...", "plz
explain...") measurably lowers both the hybrid search ranking and the
cross-encoder's relevance score for the correct chunk, sometimes enough
to drop it below the grounding threshold, even though the same
question asked "cleanly" retrieves it with high confidence.

The rewrite is used only for retrieval (embedding + BM25 + reranking);
the original question is still what's shown to the generation LLM and
answered, so the final answer stays faithful to what the user actually
asked.
"""

from __future__ import annotations

import logging

from app.core.config import get_settings
from app.llm import LLMError
from app.llm import router as llm_router

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Rewrite the user's message as a short, clean, keyword-focused search "
    "query for retrieving passages from a document corpus. Preserve every "
    "substantive keyword (names, titles, technical terms). Strip filler, "
    "greetings, and politeness wrapping (e.g. 'can you', 'please', 'tell me "
    "about'). Expand informal abbreviations (e.g. 'u' -> 'you', 'plz' -> "
    "'please', 'abt' -> 'about', 'wut' -> 'what'). Reply with ONLY the "
    "rewritten query, nothing else."
)


def rewrite_for_retrieval(question: str) -> str:
    """Return a retrieval-optimized rewrite of `question`, or `question`
    unchanged if expansion is disabled or the LLM call fails.
    """
    settings = get_settings()
    if not settings.enable_query_expansion:
        return question

    try:
        rewritten = llm_router.generate_answer(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
    except LLMError as exc:
        logger.warning("Query rewrite failed, using the original question: %s", exc)
        return question

    rewritten = rewritten.strip().strip('"')
    return rewritten or question
