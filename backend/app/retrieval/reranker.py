"""Stage 7: Rerank.

Local cross-encoder reranker that scores (query, chunk) pairs jointly
for higher precision than the bi-encoder/BM25 fusion score alone.

It also fixes a score-stability problem: Weaviate's hybrid score is
min-max normalized over the candidate window, so the same query's
scores shift depending on how many candidates were requested, making
them unsafe to compare against a fixed threshold. The cross-encoder
score does not depend on the candidate pool size, so it's what the
grounding gate in app.agent.graph compares against the threshold.
"""

from __future__ import annotations

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.retrieval.weaviate_client import SearchResult


@lru_cache
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model_name)


def rerank(query: str, chunks: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Re-score `chunks` against `query` with a cross-encoder and return
    the best `top_k`, sorted descending, with each chunk's `score`
    replaced by the cross-encoder relevance score (sigmoid-normalized
    to [0, 1]; the raw model output is an unbounded logit).
    """
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, chunk.chunk_text) for chunk in chunks]
    raw_scores = model.predict(pairs, convert_to_numpy=True)

    for chunk, raw_score in zip(chunks, raw_scores):
        chunk.score = 1.0 / (1.0 + math.exp(-float(raw_score)))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_k]
