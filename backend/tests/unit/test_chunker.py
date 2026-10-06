"""Unit tests for app.ingestion.chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk, _count_tokens, chunk_document
from app.ingestion.loaders import LoadedDocument, PageText


def _doc(pages: list[PageText], filename: str = "doc.txt") -> LoadedDocument:
    return LoadedDocument(filename=filename, file_type="txt", pages=pages)


def test_chunks_never_exceed_chunk_size():
    long_text = " ".join(f"word{i}." for i in range(500))
    document = _doc([PageText(page_number=None, text=long_text)])

    chunks = chunk_document(document, doc_id="d1", chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 50


def test_chunk_overlap_carries_trailing_words_into_next_chunk():
    long_text = " ".join(f"word{i}." for i in range(200))
    document = _doc([PageText(page_number=None, text=long_text)])

    chunks = chunk_document(document, doc_id="d1", chunk_size=30, chunk_overlap=8)

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap = first_words[-8:]
    assert second_words[: len(overlap)] == overlap


def test_chunk_metadata_fields_are_set_and_sequential_across_pages():
    pages = [
        PageText(page_number=1, text="First page sentence one. First page sentence two."),
        PageText(page_number=2, text="Second page sentence one. Second page sentence two."),
    ]
    document = _doc(pages, filename="report.pdf")

    chunks = chunk_document(document, doc_id="abc-123", chunk_size=100, chunk_overlap=10)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "abc-123" for c in chunks)
    assert all(c.filename == "report.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    pages_seen = [c.page for c in chunks]
    assert 1 in pages_seen
    assert 2 in pages_seen
    # page 1's chunks must come before page 2's chunks given the index ordering
    first_page2_index = pages_seen.index(2)
    assert all(p == 1 for p in pages_seen[:first_page2_index])


def test_chunk_generates_doc_id_when_none_given():
    document = _doc([PageText(page_number=None, text="Some short text.")])

    chunks = chunk_document(document, doc_id=None, chunk_size=50, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0].doc_id  # non-empty generated uuid string


def test_chunk_empty_page_produces_no_chunks():
    document = _doc([PageText(page_number=1, text="   ")])

    chunks = chunk_document(document, doc_id="d1", chunk_size=50, chunk_overlap=5)

    assert chunks == []


def test_chunk_respects_sentence_boundaries_for_small_input():
    text = "Short sentence one. Short sentence two. Short sentence three."
    document = _doc([PageText(page_number=None, text=text)])

    chunks = chunk_document(document, doc_id="d1", chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text.strip().endswith("Short sentence three.")


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (0, 0),
        (-5, 0),
        (50, 50),
        (50, 60),
    ],
)
def test_chunk_invalid_size_or_overlap_raises_value_error(chunk_size, chunk_overlap):
    document = _doc([PageText(page_number=None, text="Some text.")])

    with pytest.raises(ValueError):
        chunk_document(document, doc_id="d1", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_hard_splits_a_single_oversized_sentence():
    huge_sentence = " ".join(f"w{i}" for i in range(300)) + "."
    document = _doc([PageText(page_number=None, text=huge_sentence)])

    chunks = chunk_document(document, doc_id="d1", chunk_size=50, chunk_overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 50
