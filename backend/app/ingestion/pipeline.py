"""Orchestrates load -> chunk -> embed -> upsert for a single file already
on disk. Shared by the API upload endpoint (app/api/collections.py) and
the sample-seeding script (scripts/seed_samples.py) so both follow
exactly the same ingest logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    compute_file_hash,
    load_document,
)
from app.retrieval import weaviate_client as vector_store


@dataclass
class IngestResult:
    filename: str
    success: bool
    skipped: bool = False
    doc_id: str | None = None
    chunk_count: int | None = None
    error: str | None = None


def ingest_file(collection: str, path: Path, *, original_filename: str | None = None) -> IngestResult:
    """Load, chunk, and upsert the file at `path` into `collection`.

    `original_filename` overrides the name recorded on each chunk (and
    therefore shown back by GET .../documents) when `path`'s own name
    isn't the one a user/caller should see, e.g. when the caller stored
    the upload on disk under a content-hash filename.

    Never raises for expected failure modes (unsupported type, corrupt/
    empty file, vector-store error) -- those come back as a failed
    IngestResult so a caller processing a batch of files can continue.
    """
    filename = original_filename or path.name
    settings = get_settings()

    try:
        content = path.read_bytes()
        doc_id = compute_file_hash(content)

        document = load_document(path)
        document.filename = filename

        chunks = chunk_document(
            document,
            doc_id=doc_id,
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
        )
        add_result = vector_store.add_documents(collection, chunks)

        return IngestResult(
            filename=filename,
            success=True,
            skipped=add_result.skipped,
            doc_id=doc_id,
            chunk_count=len(chunks),
        )
    except (
        UnsupportedFileTypeError,
        DocumentLoadError,
        EmptyDocumentError,
        vector_store.VectorStoreError,
    ) as exc:
        return IngestResult(filename=filename, success=False, error=str(exc))
