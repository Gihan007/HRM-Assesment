"""Data structures returned by the RAG query pipeline: a citation entry
and the overall result (answer, citations, retrieved chunks for debugging).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.weaviate_client import SearchResult

NOT_FOUND_MESSAGE = "I couldn't find this in the provided documents."
CHITCHAT_MESSAGE = (
    "Hi! I'm a document Q&A assistant for this collection — ask me a question "
    "about the documents in it and I'll answer using them."
)


@dataclass
class Citation:
    index: int  # 1-based, matches the [n] marker used in the answer text
    filename: str
    page: int | None
    snippet: str


@dataclass
class QueryResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[SearchResult]
