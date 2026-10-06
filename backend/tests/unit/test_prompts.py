"""Unit tests for app.agent.prompts."""

from __future__ import annotations

from app.agent.prompts import SYSTEM_PROMPT, build_messages
from app.retrieval.weaviate_client import SearchResult


def _chunk(text, filename="doc.txt", page=None, doc_id="d1", chunk_index=0, score=0.9):
    return SearchResult(
        chunk_text=text,
        doc_id=doc_id,
        filename=filename,
        page=page,
        chunk_index=chunk_index,
        score=score,
    )


def test_build_messages_includes_system_and_user_turn():
    chunks = [_chunk("Paris is the capital of France.")]
    messages = build_messages("What is the capital of France?", chunks)

    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"


def test_build_messages_numbers_sources_in_order():
    chunks = [
        _chunk("First fact.", filename="a.txt"),
        _chunk("Second fact.", filename="b.txt"),
    ]
    user_content = build_messages("A question?", chunks)[1]["content"]

    assert "[1] (a.txt)" in user_content
    assert "[2] (b.txt)" in user_content
    assert user_content.index("[1]") < user_content.index("[2]")


def test_build_messages_includes_page_number_when_present():
    chunks = [_chunk("Some fact.", filename="report.pdf", page=5)]
    user_content = build_messages("A question?", chunks)[1]["content"]

    assert "report.pdf, page 5" in user_content


def test_build_messages_omits_page_when_absent():
    chunks = [_chunk("Some fact.", filename="notes.txt", page=None)]
    user_content = build_messages("A question?", chunks)[1]["content"]

    assert "notes.txt, page" not in user_content


def test_build_messages_includes_question_and_chunk_text():
    chunks = [_chunk("The sky is blue.")]
    user_content = build_messages("What color is the sky?", chunks)[1]["content"]

    assert "What color is the sky?" in user_content
    assert "The sky is blue." in user_content
