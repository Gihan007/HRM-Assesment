"""System/grounding prompt template for the RAG generate step.

Kept in its own module, separate from the pipeline orchestration, so
the exact wording sent to the LLM is easy to find and tune without
touching retrieval/control-flow code.
"""

from __future__ import annotations

from app.retrieval.weaviate_client import SearchResult

SYSTEM_PROMPT = (
    "You are a RAG assistant. Answer the user's question using ONLY the "
    "numbered sources below.\n"
    "- Cite every factual claim with the matching bracket number(s), e.g. [1] or [1][3].\n"
    "- Do not use any knowledge beyond what the sources say.\n"
    "- If the sources do not contain enough information to answer, say so "
    "explicitly instead of guessing.\n"
    "- Be concise and answer the question directly."
)


def _format_source(index: int, chunk: SearchResult) -> str:
    location = chunk.filename
    if chunk.page is not None:
        location += f", page {chunk.page}"
    return f"[{index}] ({location})\n{chunk.chunk_text}"


def build_messages(question: str, chunks: list[SearchResult]) -> list[dict]:
    """Build the chat messages for the generate step: a system prompt
    plus a user turn with numbered sources and the question.
    """
    sources_block = "\n\n".join(_format_source(i, chunk) for i, chunk in enumerate(chunks, start=1))
    user_content = (
        f"Sources:\n{sources_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, with inline [n] citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
