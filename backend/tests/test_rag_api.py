from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
from app.rag.rag_service import RAGService
from app.exceptions.rag_exceptions import (
    LLMError,
    RetrievalError,
)
client = TestClient(app)


def test_rag_query_endpoint_exists():
    response = client.post(
        "/rag/query",
        json={}
    )
    assert response.status_code == 422
    assert response.status_code != 404

def test_rag_query_empty_query():
    response = client.post(
        "/rag/query",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422

def test_rag_query_invalid_top_k():
    response = client.post(
        "/rag/query",
        json={
            "query": "What is RAG?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

def test_rag_query_top_k_too_large():
    response = client.post(
        "/rag/query",
        json={
            "query": "What is RAG?",
            "top_k": 21,
        },
    )

    assert response.status_code == 422

def test_rag_query_whitespace_only_query():
    response = client.post(
        "/rag/query",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 422

def test_rag_query_retrieval_failure():
    with patch(
        "app.api.rag.create_rag_service"
    ) as mock_create_service:

        mock_service = mock_create_service.return_value

        mock_service.generate.side_effect = RetrievalError(
            "Retrieval service failed"
        )

        response = client.post(
            "/rag/query",
            json={
                "query": "What is RAG?",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to retrieve relevant document context."
    }
    assert "Retrieval service failed" not in response.text  

def test_rag_query_llm_failure():
    with patch(
        "app.api.rag.create_rag_service"
    ) as mock_create_service:

        mock_service = mock_create_service.return_value

        mock_service.generate.side_effect = LLMError(
            "LLM provider failed"
        )

        response = client.post(
            "/rag/query",
            json={
                "query": "What is machine learning?",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to generate a response from the language model."
    }
    assert "LLM provider failed" not in response.text

def test_rag_query_success_response_contract():
    mock_response = {
        "answer": "RAG combines retrieval with generation.",
        "sources": [
            {
                "document_id": "doc-001",
                "document_name": "rag.pdf",
                "chunk_id": 1,
                "score": 0.95,
            }
        ],
        "metadata": {
            "model_name": "fake-llm",
            "retrieved_chunks": 1,
        },
    }

    with patch(
        "app.api.rag.create_rag_service"
    ) as mock_create_service:

        mock_service = mock_create_service.return_value
        mock_service.generate.return_value = mock_response

        response = client.post(
            "/rag/query",
            json={
                "query": "What is RAG?",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "metadata" in data

    assert data["answer"] == "RAG combines retrieval with generation."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc-001"
    assert data["metadata"]["model_name"] == "fake-llm"