import pytest
from pydantic import ValidationError

from app.schemas.rag_schema import (
    RAGRequest,
    RAGResponse,
    SourceReference,
)


def test_rag_request_defaults():

    request = RAGRequest(
        query="What is machine learning?"
    )

    assert request.query == "What is machine learning?"
    assert request.top_k == 5
    assert request.document_id is None


def test_rag_request_accepts_document_filter():

    request = RAGRequest(
        query="What is the leave policy?",
        top_k=3,
        document_id="doc-001",
    )

    assert request.top_k == 3
    assert request.document_id == "doc-001"


def test_rag_request_rejects_invalid_top_k():

    with pytest.raises(ValidationError):

        RAGRequest(
            query="Test query",
            top_k=0,
        )


def test_source_reference():

    source = SourceReference(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=2,
        score=0.91,
    )

    assert source.document_id == "doc-001"
    assert source.document_name == "test.pdf"
    assert source.chunk_id == 2
    assert source.score == 0.91


def test_rag_response():

    source = SourceReference(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=2,
        score=0.91,
    )

    response = RAGResponse(
        answer="Machine learning allows systems to learn from data.",
        sources=[source],
    )

    assert (
        response.answer
        == "Machine learning allows systems to learn from data."
    )

    assert len(response.sources) == 1

    assert response.sources[0].document_name == "test.pdf"


def test_rag_response_without_sources():

    response = RAGResponse(
        answer="I could not find relevant information."
    )

    assert response.sources == []
    assert response.metadata == {}


def test_rag_request_requires_query():

    with pytest.raises(ValidationError):

        RAGRequest(
            top_k=5,
        )