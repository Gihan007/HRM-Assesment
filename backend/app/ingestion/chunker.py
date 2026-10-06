"""Splits extracted text into overlapping chunks with doc/page/index metadata.

Chunking is paragraph- and sentence-aware: text is first split on blank
lines (paragraphs), then on sentence boundaries, and those pieces are
greedily packed into chunks of at most `chunk_size` tokens, carrying the
trailing `chunk_overlap` tokens of each chunk into the next one. A single
sentence longer than `chunk_size` is hard-split by word count as a last
resort so no chunk ever exceeds the configured size.

"Tokens" here means whitespace-separated words, not a model-specific
tokenizer count — an intentional approximation for chunk-size budgeting.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.ingestion.loaders import LoadedDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str


def _count_tokens(text: str) -> int:
    return len(text.split())


def _split_into_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(paragraph.strip()) if s]


def _hard_split(segment: str, chunk_size: int) -> list[str]:
    """Last-resort split of a single oversized segment by raw word count."""
    words = segment.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _build_segments(text: str, chunk_size: int) -> list[str]:
    segments: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _split_into_sentences(paragraph):
            if _count_tokens(sentence) > chunk_size:
                segments.extend(_hard_split(sentence, chunk_size))
            else:
                segments.append(sentence)
    return segments


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text.strip():
        return []

    segments = _build_segments(text, chunk_size)
    if not segments:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = _count_tokens(segment)
        if current and current_tokens + seg_tokens > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-chunk_overlap:] if chunk_overlap > 0 else []
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)
        current.append(segment)
        current_tokens += seg_tokens

    joined = " ".join(current).strip()
    if joined:
        chunks.append(joined)

    return chunks


def chunk_document(
    document: LoadedDocument,
    doc_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk every page of `document` into a flat, sequentially-indexed list.

    `chunk_index` is a single counter running across all pages of the
    document (not reset per page), so it can serve as a stable ordering
    key for the whole document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    resolved_doc_id = doc_id or str(uuid.uuid4())
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        for piece in _split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    doc_id=resolved_doc_id,
                    filename=document.filename,
                    page=page.page_number,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )
            chunk_index += 1

    return chunks
