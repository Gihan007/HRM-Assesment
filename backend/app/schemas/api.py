"""Request/response models for the public API (collections, documents,
query, health).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class CollectionResponse(BaseModel):
    name: str
    document_count: int


class DocumentUploadResult(BaseModel):
    filename: str
    success: bool
    skipped: bool = False
    doc_id: str | None = None
    chunk_count: int | None = None
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    results: list[DocumentUploadResult]


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceResponse(BaseModel):
    index: int
    filename: str
    page: int | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str
