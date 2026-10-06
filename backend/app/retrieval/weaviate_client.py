"""Thin wrapper around the Weaviate persistent client: collection CRUD,
batched embed-and-upsert, and plain vector similarity search.

Embedding is done locally (app.ingestion.embedder, sentence-transformers)
rather than by a Weaviate-side vectorizer module, so every collection is
created with self-provided vectors (Configure.Vectors.self_provided) and
vectors are always supplied explicitly on insert and on query.

Connects to a Weaviate Cloud (WCD) cluster via WEAVIATE_URL (the full
https:// REST endpoint) + WEAVIATE_API_KEY. This is a process-wide,
lazily-created singleton client (see `get_client`).
"""

from __future__ import annotations

import atexit
import re
from dataclasses import dataclass

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateClient
from weaviate.collections.classes.data import DataObject

from app.core.config import get_settings
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_query, embed_texts

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")
_REPEATED_UNDERSCORES = re.compile(r"_+")
_MAX_COLLECTION_NAME_LENGTH = 63  # Weaviate class name limit


class VectorStoreError(Exception):
    """Raised when a vector-store operation fails in a way the caller
    should handle explicitly (bad input, partial batch failure, etc.)."""


@dataclass
class SearchResult:
    chunk_text: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float


@dataclass
class AddDocumentsResult:
    inserted_count: int
    skipped: bool  # True when doc_id already existed and nothing was inserted


_client: WeaviateClient | None = None


def get_client() -> WeaviateClient:
    """Return a process-wide Weaviate client, connecting lazily on first use."""
    global _client
    if _client is None or not _client.is_connected():
        settings = get_settings()
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=Auth.api_key(settings.weaviate_api_key),
        )
    return _client


def close_client() -> None:
    """Close the singleton client connection, if open. Mainly for test teardown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


atexit.register(close_client)


def sanitize_collection_name(name: str) -> str:
    """Turn an arbitrary string into a valid Weaviate collection name.

    Weaviate class names must start with an uppercase letter and may only
    contain letters, digits, and underscores. Invalid characters are
    replaced with underscores, repeated underscores are collapsed, and a
    name that doesn't start with a letter is prefixed so it does.
    """
    if not name or not name.strip():
        raise ValueError("Collection name must not be empty.")

    sanitized = _INVALID_NAME_CHARS.sub("_", name.strip())
    sanitized = _REPEATED_UNDERSCORES.sub("_", sanitized).strip("_")

    if not sanitized:
        raise ValueError(f"Collection name '{name}' has no valid characters after sanitization.")

    if not sanitized[0].isalpha():
        sanitized = f"C_{sanitized}"

    sanitized = sanitized[0].upper() + sanitized[1:]

    return sanitized[:_MAX_COLLECTION_NAME_LENGTH]


def create_collection(name: str) -> str:
    """Create a collection if it doesn't already exist.

    Returns the sanitized name actually used. Idempotent.
    """
    sanitized = sanitize_collection_name(name)
    client = get_client()

    if client.collections.exists(sanitized):
        return sanitized

    client.collections.create(
        sanitized,
        # Vectors are supplied by us (sentence-transformers), not computed by
        # Weaviate. This Weaviate Cloud sandbox cluster only allows the
        # "hfresh" vector index type (HNSW is rejected with CONFIG_NOT_ALLOWED).
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hfresh(),
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )
    return sanitized


def list_collections() -> list[str]:
    """Return the names of all existing collections."""
    client = get_client()
    return sorted(client.collections.list_all(simple=True).keys())


def delete_collection(name: str) -> None:
    """Delete a collection if it exists. Idempotent (no error if missing)."""
    sanitized = sanitize_collection_name(name)
    client = get_client()
    if client.collections.exists(sanitized):
        client.collections.delete(sanitized)


def document_exists(collection: str, doc_id: str) -> bool:
    """Check whether any chunk for `doc_id` is already present in
    `collection`. Used by `add_documents` to avoid re-ingesting the same
    file (same content hash) twice.
    """
    sanitized = sanitize_collection_name(collection)
    client = get_client()
    if not client.collections.exists(sanitized):
        return False

    coll = client.collections.get(sanitized)
    result = coll.query.fetch_objects(
        filters=Filter.by_property("doc_id").equal(doc_id),
        limit=1,
    )
    return len(result.objects) > 0


def add_documents(
    collection: str,
    chunks: list[Chunk],
    *,
    batch_size: int = 64,
    skip_if_doc_exists: bool = True,
) -> AddDocumentsResult:
    """Embed `chunks` (batched, via the configured sentence-transformers
    model) and upsert them into `collection`, creating it first if needed.

    All chunks in `chunks` are expected to belong to the same document
    (same `doc_id`, produced by `app.ingestion.chunker.chunk_document`).
    If `skip_if_doc_exists` is True (the default) and a chunk with that
    doc_id is already stored, the whole batch is skipped rather than
    inserting duplicate chunks for a file that was already ingested.
    """
    if not chunks:
        raise VectorStoreError("add_documents called with an empty chunk list.")

    sanitized = create_collection(collection)

    doc_id = chunks[0].doc_id
    if skip_if_doc_exists and document_exists(sanitized, doc_id):
        return AddDocumentsResult(inserted_count=0, skipped=True)

    client = get_client()
    coll = client.collections.get(sanitized)

    inserted = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([c.text for c in batch])

        objects = []
        for chunk, vector in zip(batch, vectors):
            properties = {
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page is not None:
                properties["page"] = chunk.page
            objects.append(DataObject(properties=properties, vector=vector))

        response = coll.data.insert_many(objects)
        if response.has_errors:
            details = "; ".join(str(err) for err in response.errors.values())
            raise VectorStoreError(
                f"Failed to insert {len(response.errors)} chunk(s) into '{sanitized}': {details}"
            )
        inserted += len(objects)

    return AddDocumentsResult(inserted_count=inserted, skipped=False)


def search(collection: str, query: str, top_k: int = 5) -> list[SearchResult]:
    """Embed `query` and return the top_k most similar chunks in
    `collection`, each with its metadata and a similarity score in
    [0, 1] (1 = identical, derived from cosine distance where available).
    """
    sanitized = sanitize_collection_name(collection)
    client = get_client()

    if not client.collections.exists(sanitized):
        raise VectorStoreError(f"Collection '{sanitized}' does not exist.")

    coll = client.collections.get(sanitized)
    query_vector = embed_query(query)

    response = coll.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True, certainty=True),
    )

    results: list[SearchResult] = []
    for obj in response.objects:
        props = obj.properties
        distance = obj.metadata.distance
        certainty = obj.metadata.certainty
        if distance is not None:
            score = 1.0 - distance
        elif certainty is not None:
            score = certainty
        else:
            score = 0.0

        results.append(
            SearchResult(
                chunk_text=props.get("text", ""),
                doc_id=props.get("doc_id", ""),
                filename=props.get("filename", ""),
                page=props.get("page"),
                chunk_index=props.get("chunk_index", 0),
                score=score,
            )
        )

    return results
