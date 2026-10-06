"""Unit tests for app.agent.graph.answer_query: the end-to-end RAG query
pipeline (retrieve -> threshold gate -> prompt -> generate -> citations).

Both the vector store search and the LLM call are mocked so these tests
run offline with no network access and no real model calls.
"""

from __future__ import annotations

import pytest

from app.agent import graph
from app.agent.state import NOT_FOUND_MESSAGE
from app.llm import LLMRequestError
from app.retrieval.weaviate_client import SearchResult


def _chunk(text, filename="doc.txt", page=None, chunk_index=0, score=0.9, doc_id="d1"):
    return SearchResult(
        chunk_text=text,
        doc_id=doc_id,
        filename=filename,
        page=page,
        chunk_index=chunk_index,
        score=score,
    )


def test_answer_query_skips_llm_when_best_score_below_threshold(monkeypatch):
    low_score_chunks = [_chunk("Irrelevant content.", score=0.1)]
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: low_score_chunks)

    def fail_if_called(messages, **kw):
        raise AssertionError("LLM should not be called when below threshold")

    monkeypatch.setattr(graph.llm_router, "generate_answer", fail_if_called)

    result = graph.answer_query("MyCollection", "irrelevant question?", similarity_threshold=0.35)

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    assert result.retrieved_chunks == low_score_chunks


def test_answer_query_skips_llm_when_no_chunks_retrieved(monkeypatch):
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: [])

    def fail_if_called(messages, **kw):
        raise AssertionError("LLM should not be called with no chunks")

    monkeypatch.setattr(graph.llm_router, "generate_answer", fail_if_called)

    result = graph.answer_query("MyCollection", "anything?")

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.retrieved_chunks == []


def test_answer_query_returns_grounded_answer_with_mapped_citations(monkeypatch):
    chunks = [
        _chunk("The Eiffel Tower is in Paris.", filename="landmarks.txt", page=None, chunk_index=0, score=0.9),
        _chunk("The Eiffel Tower was completed in 1889.", filename="history.txt", page=2, chunk_index=3, score=0.8),
    ]
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: chunks)
    monkeypatch.setattr(
        graph.llm_router,
        "generate_answer",
        lambda messages, **kw: "The Eiffel Tower is in Paris [1] and was completed in 1889 [2].",
    )

    result = graph.answer_query("MyCollection", "Tell me about the Eiffel Tower.")

    assert result.answer == "The Eiffel Tower is in Paris [1] and was completed in 1889 [2]."
    assert len(result.citations) == 2
    assert result.citations[0].index == 1
    assert result.citations[0].filename == "landmarks.txt"
    assert result.citations[0].page is None
    assert result.citations[1].index == 2
    assert result.citations[1].filename == "history.txt"
    assert result.citations[1].page == 2
    assert result.retrieved_chunks == chunks


def test_answer_query_drops_out_of_range_citation_markers(monkeypatch):
    chunks = [_chunk("Only one source.", score=0.9)]
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: chunks)
    monkeypatch.setattr(
        graph.llm_router,
        "generate_answer",
        lambda messages, **kw: "Here is the answer [1], and also [5] which doesn't exist.",
    )

    result = graph.answer_query("MyCollection", "a question?")

    assert len(result.citations) == 1
    assert result.citations[0].index == 1


def test_answer_query_handles_llm_failure_gracefully(monkeypatch):
    chunks = [_chunk("Some relevant content.", score=0.9)]
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: chunks)

    def failing_generate(messages, **kw):
        raise LLMRequestError("both providers down")

    monkeypatch.setattr(graph.llm_router, "generate_answer", failing_generate)

    result = graph.answer_query("MyCollection", "a question?")

    assert "could not be generated" in result.answer.lower()
    assert result.citations == []
    assert result.retrieved_chunks == chunks


def test_answer_query_truncates_long_citation_snippets(monkeypatch):
    long_text = "word " * 200
    chunks = [_chunk(long_text, score=0.9)]
    monkeypatch.setattr(graph, "vector_search", lambda collection, query, top_k: chunks)
    monkeypatch.setattr(graph.llm_router, "generate_answer", lambda messages, **kw: "Answer [1].")

    result = graph.answer_query("MyCollection", "a question?")

    assert len(result.citations[0].snippet) <= graph._SNIPPET_MAX_LENGTH
    assert result.citations[0].snippet.endswith("…")
