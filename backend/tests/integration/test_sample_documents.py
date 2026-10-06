"""Integration test: load + chunk the real sample PDFs shipped in
samples/ai_papers/ (not fixtures) to make sure the ingestion pipeline
works end-to-end against real, non-trivial documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import load_document

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = REPO_ROOT / "samples" / "ai_papers"

pdf_paths = sorted(SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.exists() else []


@pytest.mark.skipif(not pdf_paths, reason="samples/ai_papers has no PDFs to test against")
@pytest.mark.parametrize("pdf_path", pdf_paths, ids=lambda p: p.name)
def test_real_sample_pdf_loads_and_chunks(pdf_path):
    document = load_document(pdf_path)

    assert document.file_type == "pdf"
    assert len(document.pages) > 0
    assert all(page.page_number == i + 1 for i, page in enumerate(document.pages))

    chunks = chunk_document(document, doc_id=pdf_path.stem, chunk_size=800, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.filename == pdf_path.name for chunk in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
