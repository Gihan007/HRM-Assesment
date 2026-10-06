"""Local sentence-transformers embedding model wrapper.

The model is loaded once per process (lru_cache) since loading it is
expensive; `embed_texts`/`embed_query` are the functions the rest of the
app should call rather than touching the model directly, so the model
choice stays swappable via EMBEDDING_MODEL_NAME without other code
changes.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts. Order of the output matches `texts`."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
