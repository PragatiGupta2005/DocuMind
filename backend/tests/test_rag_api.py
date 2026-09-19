from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
from app.rag.rag_service import RAGService

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

        mock_service.generate.side_effect = RuntimeError(
            "Retrieval service failed"
        )

        response = client.post(
            "/rag/query",
            json={
                "query": "What is RAG?",
            },
        )

    assert response.status_code == 500  