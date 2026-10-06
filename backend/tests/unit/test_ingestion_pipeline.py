"""Unit tests for app.ingestion.pipeline.ingest_file.

vector_store.add_documents is mocked so these run offline; loading and
chunking use the real fixture files.
"""

from __future__ import annotations

from app.ingestion import pipeline
from app.retrieval.weaviate_client import AddDocumentsResult


def test_ingest_file_success_reports_chunk_count(monkeypatch, fixtures_dir):
    monkeypatch.setattr(
        pipeline.vector_store,
        "add_documents",
        lambda collection, chunks: AddDocumentsResult(inserted_count=len(chunks), skipped=False),
    )

    result = pipeline.ingest_file("MyCollection", fixtures_dir / "sample.txt")

    assert result.success is True
    assert result.skipped is False
    assert result.chunk_count == 1
    assert result.filename == "sample.txt"
    assert result.doc_id


def test_ingest_file_uses_original_filename_override(monkeypatch, fixtures_dir):
    monkeypatch.setattr(
        pipeline.vector_store,
        "add_documents",
        lambda collection, chunks: AddDocumentsResult(inserted_count=len(chunks), skipped=False),
    )

    result = pipeline.ingest_file(
        "MyCollection", fixtures_dir / "sample.txt", original_filename="user_upload.txt"
    )

    assert result.filename == "user_upload.txt"


def test_ingest_file_reports_skipped_when_already_ingested(monkeypatch, fixtures_dir):
    monkeypatch.setattr(
        pipeline.vector_store,
        "add_documents",
        lambda collection, chunks: AddDocumentsResult(inserted_count=0, skipped=True),
    )

    result = pipeline.ingest_file("MyCollection", fixtures_dir / "sample.txt")

    assert result.success is True
    assert result.skipped is True


def test_ingest_file_handles_unsupported_type_without_raising(fixtures_dir):
    result = pipeline.ingest_file("MyCollection", fixtures_dir / "unsupported.xyz")

    assert result.success is False
    assert result.error


def test_ingest_file_handles_corrupt_file_without_raising(fixtures_dir):
    result = pipeline.ingest_file("MyCollection", fixtures_dir / "corrupt.pdf")

    assert result.success is False
    assert result.error


def test_ingest_file_handles_empty_file_without_raising(fixtures_dir):
    result = pipeline.ingest_file("MyCollection", fixtures_dir / "empty.txt")

    assert result.success is False
    assert result.error
