"""Integration tests for app.retrieval.weaviate_client against a real,
live Weaviate Cloud cluster (WEAVIATE_URL/WEAVIATE_API_KEY from .env).

Skipped automatically if no API key is configured, so the unit test
suite stays runnable offline / without credentials.
"""

from __future__ import annotations

import time
import uuid

import pytest

from app.core.config import get_settings
from app.ingestion.chunker import Chunk
from app.retrieval import weaviate_client as wc

pytestmark = pytest.mark.skipif(
    not get_settings().weaviate_api_key,
    reason="WEAVIATE_API_KEY not configured; skipping live Weaviate Cloud tests.",
)


@pytest.fixture
def test_collection_name():
    """A unique collection name per test run, cleaned up afterward."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield name
    wc.delete_collection(name)


def _make_chunks(doc_id: str, texts: list[str], filename: str = "doc.txt") -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, filename=filename, page=None, chunk_index=i, text=text)
        for i, text in enumerate(texts)
    ]


def _search_with_retry(collection, query, top_k, expected_count=1, attempts=5, delay=1.0):
    """The cluster's `hfresh` vector index builds asynchronously; under
    rapid back-to-back collection creates (as in this test suite) a
    just-inserted chunk can briefly be unsearchable. Poll instead of
    asserting immediately.
    """
    for attempt in range(attempts):
        results = wc.search(collection, query, top_k=top_k)
        if len(results) >= expected_count:
            return results
        if attempt < attempts - 1:
            time.sleep(delay)
    return results


def test_create_list_delete_collection_round_trip(test_collection_name):
    created_name = wc.create_collection(test_collection_name)
    assert created_name in wc.list_collections()

    wc.delete_collection(test_collection_name)
    assert created_name not in wc.list_collections()

    # recreate so the fixture's own teardown delete is a harmless no-op
    wc.create_collection(test_collection_name)


def test_create_collection_is_idempotent(test_collection_name):
    first = wc.create_collection(test_collection_name)
    second = wc.create_collection(test_collection_name)
    assert first == second


def test_add_documents_and_search_returns_relevant_chunk(test_collection_name):
    chunks = _make_chunks(
        doc_id="doc-1",
        texts=[
            "The Eiffel Tower is located in Paris, France.",
            "Bananas are a good source of potassium.",
            "Mount Everest is the tallest mountain on Earth.",
        ],
    )

    result = wc.add_documents(test_collection_name, chunks)
    assert result.inserted_count == 3
    assert result.skipped is False

    results = _search_with_retry(test_collection_name, "Where is the Eiffel Tower?", top_k=1)
    assert len(results) == 1
    assert "Eiffel Tower" in results[0].chunk_text
    assert results[0].doc_id == "doc-1"
    assert 0.0 <= results[0].score <= 1.0


def test_add_documents_skips_duplicate_doc_id(test_collection_name):
    chunks = _make_chunks(doc_id="doc-dup", texts=["Some content about ducks."])

    first = wc.add_documents(test_collection_name, chunks)
    assert first.inserted_count == 1
    assert first.skipped is False

    second = wc.add_documents(test_collection_name, chunks)
    assert second.inserted_count == 0
    assert second.skipped is True

    # only one copy should actually be stored
    results = _search_with_retry(test_collection_name, "ducks", top_k=10)
    assert len(results) == 1


def test_search_preserves_page_and_chunk_index_metadata(test_collection_name):
    chunks = [
        Chunk(doc_id="doc-meta", filename="report.pdf", page=3, chunk_index=7, text="Quarterly revenue grew by 12 percent."),
    ]
    wc.add_documents(test_collection_name, chunks)

    results = _search_with_retry(test_collection_name, "revenue growth", top_k=1)
    assert results[0].filename == "report.pdf"
    assert results[0].page == 3
    assert results[0].chunk_index == 7


def test_search_against_missing_collection_raises():
    with pytest.raises(wc.VectorStoreError):
        wc.search("Collection_that_does_not_exist_xyz", "anything", top_k=1)


def test_add_documents_sanitizes_collection_name(test_collection_name):
    unsanitized = test_collection_name.replace("_", " ") + "!!"
    chunks = _make_chunks(doc_id="doc-sanitize", texts=["Some sanitized-name test content."])

    wc.add_documents(unsanitized, chunks)
    sanitized_name = wc.sanitize_collection_name(unsanitized)

    assert sanitized_name in wc.list_collections()
    wc.delete_collection(sanitized_name)
