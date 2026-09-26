from app.core.settings import (
    RAG_DEFAULT_TOP_K,
    RAG_MAX_TOP_K,
    MIN_RELEVANCE_SCORE,
)
from app.schemas.rag_schema import RAGRequest
import pytest

def test_rag_default_top_k_is_configured():
    assert isinstance(RAG_DEFAULT_TOP_K, int)
    assert RAG_DEFAULT_TOP_K >= 1

def test_rag_max_top_k_is_configured():
    assert isinstance(RAG_MAX_TOP_K, int)
    assert RAG_MAX_TOP_K >= RAG_DEFAULT_TOP_K

def test_min_relevance_score_is_configured():
    assert isinstance(MIN_RELEVANCE_SCORE, float)
    assert 0.0 <= MIN_RELEVANCE_SCORE <= 1.0

def test_rag_request_uses_configured_default_top_k():
    request = RAGRequest(
        query="What is machine learning?"
    )

    assert request.top_k == RAG_DEFAULT_TOP_K

def test_rag_request_rejects_top_k_above_configured_max():
    with pytest.raises(ValueError):
        RAGRequest(
            query="What is machine learning?",
            top_k=RAG_MAX_TOP_K + 1,
        )