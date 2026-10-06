"""Collection, document ingestion, and query routes.

POST   /collections                     create a collection
GET    /collections                     list collections (with document count)
DELETE /collections/{name}              delete a collection
POST   /collections/{name}/documents    upload one or more files (multipart)
GET    /collections/{name}/documents    list ingested files
POST   /collections/{name}/query        ask a question, get a grounded answer
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.agent.graph import answer_query
from app.core.config import get_settings
from app.ingestion.loaders import compute_file_hash
from app.ingestion.pipeline import ingest_file
from app.retrieval import weaviate_client as vector_store
from app.schemas.api import (
    CollectionCreateRequest,
    CollectionResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
    DocumentUploadResult,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

router = APIRouter(prefix="/collections", tags=["collections"])


def _ensure_collection_exists(name: str) -> None:
    try:
        exists = vector_store.collection_exists(name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' does not exist.",
        )


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
def create_collection(payload: CollectionCreateRequest) -> CollectionResponse:
    try:
        already_exists = vector_store.collection_exists(payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if already_exists:
        sanitized = vector_store.sanitize_collection_name(payload.name)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection '{sanitized}' already exists.",
        )

    name = vector_store.create_collection(payload.name)
    return CollectionResponse(name=name, document_count=0)


@router.get("", response_model=list[CollectionResponse])
def list_collections() -> list[CollectionResponse]:
    results = []
    for name in vector_store.list_collections():
        document_count = len(vector_store.list_documents(name))
        results.append(CollectionResponse(name=name, document_count=document_count))
    return results


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_collection(name: str) -> None:
    _ensure_collection_exists(name)
    vector_store.delete_collection(name)


@router.post(
    "/{name}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_207_MULTI_STATUS,
)
async def upload_documents(
    name: str, files: list[UploadFile] = File(...)
) -> DocumentUploadResponse:
    """Ingest one or more files into `name`. Each file is handled
    independently: one bad/oversized/corrupt file does not abort the
    rest of the batch, and the response reports per-file success/error.
    """
    _ensure_collection_exists(name)

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    upload_dir = Path(settings.upload_dir) / vector_store.sanitize_collection_name(name)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results: list[DocumentUploadResult] = []

    for upload in files:
        original_filename = upload.filename or "unnamed"
        content = await upload.read()

        if len(content) > max_bytes:
            size_mb = len(content) / (1024 * 1024)
            results.append(
                DocumentUploadResult(
                    filename=original_filename,
                    success=False,
                    error=(
                        f"File too large ({size_mb:.1f} MB); limit is {settings.max_upload_mb} MB."
                    ),
                )
            )
            continue

        suffix = Path(original_filename).suffix
        stored_path = upload_dir / f"{compute_file_hash(content)}{suffix}"
        stored_path.write_bytes(content)

        ingest_result = ingest_file(name, stored_path, original_filename=original_filename)

        results.append(
            DocumentUploadResult(
                filename=ingest_result.filename,
                success=ingest_result.success,
                skipped=ingest_result.skipped,
                doc_id=ingest_result.doc_id,
                chunk_count=ingest_result.chunk_count,
                error=ingest_result.error,
            )
        )

    return DocumentUploadResponse(results=results)


@router.get("/{name}/documents", response_model=list[DocumentSummaryResponse])
def list_documents(name: str) -> list[DocumentSummaryResponse]:
    _ensure_collection_exists(name)
    documents = vector_store.list_documents(name)
    return [
        DocumentSummaryResponse(doc_id=d.doc_id, filename=d.filename, chunk_count=d.chunk_count)
        for d in documents
    ]


@router.post("/{name}/query", response_model=QueryResponse)
def query_collection(name: str, payload: QueryRequest) -> QueryResponse:
    _ensure_collection_exists(name)

    result = answer_query(
        name,
        payload.question,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
    )

    sources = [
        SourceResponse(index=c.index, filename=c.filename, page=c.page, snippet=c.snippet)
        for c in result.citations
    ]
    return QueryResponse(answer=result.answer, sources=sources)
