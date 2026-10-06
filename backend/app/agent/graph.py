"""RAG query pipeline: retrieve -> grounding threshold gate -> prompt ->
generate -> citation mapping.

This is currently a plain function pipeline, not yet a LangGraph
StateGraph: it's the "retrieve + generate" core described in
plan_recorrected.md, callable directly. A supervisor node with real
tool-calling/routing will wrap `answer_query` as a LangGraph node once
that's needed; the function boundary is kept stable so that wrapping
doesn't require changing this logic.
"""

from __future__ import annotations

import logging
import re

from app.agent.prompts import build_messages
from app.agent.state import NOT_FOUND_MESSAGE, Citation, QueryResult
from app.core.config import get_settings
from app.llm import LLMError
from app.llm import router as llm_router
from app.retrieval.weaviate_client import search as vector_search

logger = logging.getLogger(__name__)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")
_SNIPPET_MAX_LENGTH = 300

_LLM_UNAVAILABLE_MESSAGE = (
    "The answer could not be generated right now (the LLM provider is "
    "unavailable). Please try again shortly."
)


def answer_query(
    collection: str,
    question: str,
    *,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> QueryResult:
    """Run the full query pipeline and return a grounded, cited answer.

    1. Retrieve `top_k` chunks from `collection`.
    2. If the best chunk's similarity score is below `similarity_threshold`,
       return NOT_FOUND_MESSAGE without calling the LLM at all.
    3. Otherwise build a numbered-sources prompt and generate an answer
       (Groq primary, OpenRouter fallback), instructing the model to
       answer only from the sources and cite them inline as [n].
    4. Map cited [n] markers back to real chunk metadata for the
       `citations` list; marker numbers the model invents (out of range)
       are dropped rather than surfaced as a broken reference.

    LLM/API failures (missing key, rate limit, network error) are caught
    and turned into a plain-language answer rather than raising.
    """
    settings = get_settings()
    resolved_top_k = top_k if top_k is not None else settings.top_k
    resolved_threshold = (
        similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    )

    chunks = vector_search(collection, question, top_k=resolved_top_k)

    if not chunks or chunks[0].score < resolved_threshold:
        return QueryResult(answer=NOT_FOUND_MESSAGE, citations=[], retrieved_chunks=chunks)

    messages = build_messages(question, chunks)

    try:
        answer_text = llm_router.generate_answer(messages)
    except LLMError as exc:
        logger.error("LLM generation failed: %s", exc)
        return QueryResult(answer=_LLM_UNAVAILABLE_MESSAGE, citations=[], retrieved_chunks=chunks)

    citations = _build_citations(answer_text, chunks)

    return QueryResult(answer=answer_text, citations=citations, retrieved_chunks=chunks)


def _build_citations(answer_text: str, chunks: list) -> list[Citation]:
    cited_indices = sorted({int(match) for match in _CITATION_MARKER.findall(answer_text)})
    return [
        Citation(
            index=i,
            filename=chunks[i - 1].filename,
            page=chunks[i - 1].page,
            snippet=_truncate(chunks[i - 1].chunk_text),
        )
        for i in cited_indices
        if 1 <= i <= len(chunks)
    ]


def _truncate(text: str, max_length: int = _SNIPPET_MAX_LENGTH) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
