"""Stages 5-6: Hybrid retrieval + retrieve wide.

Runs Weaviate hybrid search (BM25 + vector fusion) under the prefilter,
requesting a wide candidate pool (TOP_K_WIDE) for the reranker to
narrow down.

No logic implemented yet.
"""
