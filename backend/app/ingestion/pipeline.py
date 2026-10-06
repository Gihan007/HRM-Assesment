"""Orchestrates load -> chunk -> embed -> upsert into Weaviate.

Also responsible for configuring the Weaviate class so the `text`
property is indexed for BM25 (the hybrid-search prerequisite described
as Stage 0 in plan_recorrected.md).

No logic implemented yet.
"""
