"""Unit tests for the pure, network-free parts of app.retrieval.weaviate_client:
collection name sanitization. Add/search/collection-CRUD against a live
cluster are covered by tests/integration/test_weaviate_client.py.
"""

from __future__ import annotations

import pytest

from app.retrieval.weaviate_client import sanitize_collection_name


def test_sanitize_replaces_invalid_characters():
    assert sanitize_collection_name("user 123-contracts!!") == "User_123_contracts"


def test_sanitize_collapses_repeated_underscores():
    assert sanitize_collection_name("a___b") == "A_b"


def test_sanitize_prefixes_when_name_starts_with_digit():
    assert sanitize_collection_name("123abc") == "C_123abc"


def test_sanitize_capitalizes_first_letter():
    assert sanitize_collection_name("contracts_2024") == "Contracts_2024"
    assert sanitize_collection_name("Contracts_2024") == "Contracts_2024"


def test_sanitize_strips_leading_and_trailing_separators():
    assert sanitize_collection_name("  -contracts-  ") == "Contracts"


def test_sanitize_is_idempotent():
    once = sanitize_collection_name("user_42-contract set")
    twice = sanitize_collection_name(once)
    assert once == twice


def test_sanitize_truncates_overly_long_names():
    long_name = "a" * 100
    result = sanitize_collection_name(long_name)
    assert len(result) <= 63


def test_sanitize_raises_on_empty_name():
    with pytest.raises(ValueError):
        sanitize_collection_name("")


def test_sanitize_raises_when_nothing_valid_remains():
    with pytest.raises(ValueError):
        sanitize_collection_name("!!!---???")
