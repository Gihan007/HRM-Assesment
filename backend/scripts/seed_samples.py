"""Seed the default 'ai_papers' collection from samples/ai_papers/.

Run from the backend/ directory with the backend venv active:

    cd backend
    .venv/Scripts/python scripts/seed_samples.py   (Windows)
    .venv/bin/python scripts/seed_samples.py        (macOS/Linux)

Idempotent: a file already ingested (same content hash) is reported as
skipped rather than duplicated, so re-running this is always safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put backend/ on sys.path

from app.ingestion.pipeline import ingest_file  # noqa: E402
from app.retrieval import weaviate_client as vector_store  # noqa: E402

COLLECTION_NAME = "ai_papers"
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples" / "ai_papers"


def seed() -> None:
    if not SAMPLES_DIR.exists():
        print(f"No samples directory found at {SAMPLES_DIR}; nothing to seed.")
        return

    files = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_file())
    if not files:
        print(f"No files found in {SAMPLES_DIR}.")
        return

    collection = vector_store.create_collection(COLLECTION_NAME)
    print(f"Seeding collection '{collection}' from {SAMPLES_DIR}")

    for path in files:
        result = ingest_file(collection, path)
        if not result.success:
            print(f"  {path.name}: SKIPPED ({result.error})")
        elif result.skipped:
            print(f"  {path.name}: already ingested, skipped.")
        else:
            print(f"  {path.name}: ingested {result.chunk_count} chunks.")

    total_docs = len(vector_store.list_documents(collection))
    print(f"Done. '{collection}' now has {total_docs} document(s).")


if __name__ == "__main__":
    seed()
