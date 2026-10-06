"""API smoke tests for every endpoint, exercising the real FastAPI app
against the live Weaviate Cloud cluster via TestClient. No GROQ/OPENROUTER
key is configured in this environment, so query tests also double as a
check that missing-LLM-key is handled gracefully end-to-end rather than
crashing the request.

Skipped automatically if no Weaviate API key is configured.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import collections as collections_module
from app.core.config import get_settings
from app.main import app
from app.retrieval.weaviate_client import sanitize_collection_name

pytestmark = pytest.mark.skipif(
    not get_settings().weaviate_api_key,
    reason="WEAVIATE_API_KEY not configured; skipping live API smoke tests.",
)

client = TestClient(app)


@pytest.fixture
def collection_name():
    name = f"api_test_{uuid.uuid4().hex[:10]}"
    yield name
    client.delete(f"/collections/{name}")


# --- GET /health -----------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /collections -------------------------------------------------------


def test_create_collection(collection_name):
    response = client.post("/collections", json={"name": collection_name})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == sanitize_collection_name(collection_name)
    assert body["document_count"] == 0


def test_create_collection_conflict_on_duplicate(collection_name):
    client.post("/collections", json={"name": collection_name})
    response = client.post("/collections", json={"name": collection_name})

    assert response.status_code == 409


def test_create_collection_rejects_empty_name():
    response = client.post("/collections", json={"name": ""})
    assert response.status_code == 422


# --- GET /collections --------------------------------------------------------


def test_list_collections_includes_created(collection_name):
    client.post("/collections", json={"name": collection_name})

    response = client.get("/collections")

    assert response.status_code == 200
    names = [c["name"] for c in response.json()]
    assert sanitize_collection_name(collection_name) in names


# --- DELETE /collections/{name} ----------------------------------------------


def test_delete_nonexistent_collection_returns_404():
    response = client.delete("/collections/DoesNotExistXyz123")
    assert response.status_code == 404


def test_delete_collection(collection_name):
    client.post("/collections", json={"name": collection_name})

    response = client.delete(f"/collections/{collection_name}")
    assert response.status_code == 204

    response = client.get(f"/collections/{collection_name}/documents")
    assert response.status_code == 404


# --- POST /collections/{name}/documents ---------------------------------------


def test_upload_requires_existing_collection():
    response = client.post(
        "/collections/DoesNotExistXyz123/documents",
        files={"files": ("a.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 404


def test_upload_and_list_documents(collection_name, fixtures_dir):
    client.post("/collections", json={"name": collection_name})

    with open(fixtures_dir / "sample.txt", "rb") as f:
        response = client.post(
            f"/collections/{collection_name}/documents",
            files={"files": ("sample.txt", f, "text/plain")},
        )

    assert response.status_code == 207
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["success"] is True
    assert results[0]["skipped"] is False
    assert results[0]["chunk_count"] > 0

    response = client.get(f"/collections/{collection_name}/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "sample.txt"

    response = client.get("/collections")
    by_name = {c["name"]: c for c in response.json()}
    assert by_name[sanitize_collection_name(collection_name)]["document_count"] == 1


def test_upload_reports_per_file_errors_without_aborting_batch(collection_name, fixtures_dir):
    client.post("/collections", json={"name": collection_name})

    with (
        open(fixtures_dir / "sample.md", "rb") as good,
        open(fixtures_dir / "unsupported.xyz", "rb") as bad,
    ):
        response = client.post(
            f"/collections/{collection_name}/documents",
            files=[
                ("files", ("sample.md", good, "text/markdown")),
                ("files", ("unsupported.xyz", bad, "application/octet-stream")),
            ],
        )

    assert response.status_code == 207
    by_filename = {r["filename"]: r for r in response.json()["results"]}
    assert by_filename["sample.md"]["success"] is True
    assert by_filename["unsupported.xyz"]["success"] is False
    assert by_filename["unsupported.xyz"]["error"]


def test_upload_rejects_corrupt_file(collection_name, fixtures_dir):
    client.post("/collections", json={"name": collection_name})

    with open(fixtures_dir / "corrupt.pdf", "rb") as f:
        response = client.post(
            f"/collections/{collection_name}/documents",
            files={"files": ("corrupt.pdf", f, "application/pdf")},
        )

    assert response.status_code == 207
    result = response.json()["results"][0]
    assert result["success"] is False
    assert result["error"]


def test_upload_rejects_oversized_file(collection_name, monkeypatch):
    client.post("/collections", json={"name": collection_name})

    tiny_limit_settings = get_settings().model_copy(update={"max_upload_mb": 0})
    monkeypatch.setattr(collections_module, "get_settings", lambda: tiny_limit_settings)

    response = client.post(
        f"/collections/{collection_name}/documents",
        files={"files": ("sample.txt", b"this content exceeds a zero byte limit", "text/plain")},
    )

    assert response.status_code == 207
    result = response.json()["results"][0]
    assert result["success"] is False
    assert "too large" in result["error"].lower()


def test_upload_same_file_twice_is_skipped_and_not_duplicated(collection_name, fixtures_dir):
    client.post("/collections", json={"name": collection_name})
    path = fixtures_dir / "sample.txt"

    def do_upload():
        with open(path, "rb") as f:
            return client.post(
                f"/collections/{collection_name}/documents",
                files={"files": ("sample.txt", f, "text/plain")},
            )

    first = do_upload()
    second = do_upload()

    assert first.json()["results"][0]["skipped"] is False
    assert second.json()["results"][0]["skipped"] is True

    docs = client.get(f"/collections/{collection_name}/documents").json()
    assert len(docs) == 1


# --- GET /collections/{name}/documents ----------------------------------------


def test_list_documents_requires_existing_collection():
    response = client.get("/collections/DoesNotExistXyz123/documents")
    assert response.status_code == 404


def test_list_documents_empty_for_new_collection(collection_name):
    client.post("/collections", json={"name": collection_name})

    response = client.get(f"/collections/{collection_name}/documents")

    assert response.status_code == 200
    assert response.json() == []


# --- POST /collections/{name}/query -------------------------------------------


def test_query_requires_existing_collection():
    response = client.post("/collections/DoesNotExistXyz123/query", json={"question": "anything?"})
    assert response.status_code == 404


def test_query_rejects_empty_question(collection_name):
    client.post("/collections", json={"name": collection_name})

    response = client.post(f"/collections/{collection_name}/query", json={"question": ""})
    assert response.status_code == 422


def test_query_returns_not_found_when_collection_empty(collection_name):
    client.post("/collections", json={"name": collection_name})

    response = client.post(f"/collections/{collection_name}/query", json={"question": "anything?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I couldn't find this in the provided documents."
    assert body["sources"] == []


def test_query_with_ingested_content_handles_missing_llm_keys_gracefully(collection_name, fixtures_dir):
    client.post("/collections", json={"name": collection_name})
    with open(fixtures_dir / "sample.txt", "rb") as f:
        client.post(
            f"/collections/{collection_name}/documents",
            files={"files": ("sample.txt", f, "text/plain")},
        )

    response = client.post(
        f"/collections/{collection_name}/query",
        json={"question": "What does the document say about the fox?", "similarity_threshold": 0.0},
    )

    assert response.status_code == 200
    body = response.json()
    # No GROQ_API_KEY/OPENROUTER_API_KEY configured in this environment, so
    # this exercises the graceful-degradation path through the real API
    # rather than a crash or 500.
    assert "answer" in body
    assert isinstance(body["sources"], list)
