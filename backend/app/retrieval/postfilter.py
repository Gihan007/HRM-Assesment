"""Stage 8: Post-filter.

Drops chunks below RERANK_THRESHOLD, dedupes near-identical/adjacent
chunks, truncates to final TOP_K. This is also where the grounding
gate lives: if nothing survives, the generate node must skip the LLM
call and return the fixed "not in the documents" response.

No logic implemented yet.
"""
