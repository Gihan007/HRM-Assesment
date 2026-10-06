"""Unit tests for app.ingestion.loaders."""

from __future__ import annotations

from app.ingestion.loaders import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
import pytest


def test_load_txt_returns_single_unpaginated_page(fixtures_dir):
    doc = load_document(fixtures_dir / "sample.txt")

    assert doc.filename == "sample.txt"
    assert doc.file_type == "txt"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number is None
    assert "quick brown fox" in doc.pages[0].text


def test_load_md_returns_single_unpaginated_page(fixtures_dir):
    doc = load_document(fixtures_dir / "sample.md")

    assert doc.file_type == "md"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number is None
    assert "Section One" in doc.pages[0].text


def test_load_pdf_preserves_page_numbers_and_text(sample_pdf_path):
    doc = load_document(sample_pdf_path)

    assert doc.file_type == "pdf"
    assert len(doc.pages) == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert "Page one content" in doc.pages[0].text
    assert "Page two content" in doc.pages[1].text
    assert "Page three content" in doc.pages[2].text


def test_load_docx_returns_paragraph_text(sample_docx_path):
    doc = load_document(sample_docx_path)

    assert doc.file_type == "docx"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number is None
    assert "first paragraph" in doc.pages[0].text
    assert "second paragraph" in doc.pages[0].text


def test_load_empty_txt_raises_empty_document_error(fixtures_dir):
    with pytest.raises(EmptyDocumentError):
        load_document(fixtures_dir / "empty.txt")


def test_load_blank_pdf_raises_empty_document_error(empty_pdf_path):
    with pytest.raises(EmptyDocumentError):
        load_document(empty_pdf_path)


def test_load_corrupt_pdf_raises_document_load_error(fixtures_dir):
    with pytest.raises(DocumentLoadError):
        load_document(fixtures_dir / "corrupt.pdf")


def test_load_corrupt_docx_raises_document_load_error(fixtures_dir):
    with pytest.raises(DocumentLoadError):
        load_document(fixtures_dir / "corrupt.docx")


def test_load_unsupported_extension_raises_unsupported_file_type_error(fixtures_dir):
    with pytest.raises(UnsupportedFileTypeError):
        load_document(fixtures_dir / "unsupported.xyz")


def test_load_missing_file_raises_document_load_error(fixtures_dir):
    with pytest.raises(DocumentLoadError):
        load_document(fixtures_dir / "does_not_exist.txt")
