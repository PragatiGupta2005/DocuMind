from fastapi.testclient import TestClient
from app.embeddings.embedding_service import EmbeddingService
from app.main import app
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.collection_config import (
    LOCAL_COLLECTION_NAME,
)


client = TestClient(app)


def test_upload_indexes_document_in_qdrant(
    tmp_path,
    monkeypatch,
):
    """
    Verify the complete ingestion pipeline:

    Upload
    → Process
    → Chunk
    → Embed
    → Store in Qdrant
    """

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = (
        "Machine learning is a branch of artificial intelligence. "
        "Machine learning systems learn patterns from data."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "machine_learning.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    document = data["document"]
    chunks = data["chunks"]

    document_id = document["document_id"]

    assert document_id
    assert data["chunk_count"] > 0
    assert len(chunks) == data["chunk_count"]

    # Use the exact chunk UUID generated during chunking.
    chunk_uuid = chunks[0]["chunk_uuid"]

    # Connect to the real Qdrant collection.
    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    # Retrieve the exact point by its Qdrant ID.
    result = store.client.retrieve(
        collection_name=LOCAL_COLLECTION_NAME,
        ids=[chunk_uuid],
        with_payload=True,
        with_vectors=True,
    )

    # The uploaded chunk must exist in Qdrant.
    assert len(result) == 1

    point = result[0]

    # Verify point identity.
    assert str(point.id) == chunk_uuid

    # Verify document association.
    assert point.payload["document_id"] == document_id

    # Verify chunk information.
    assert point.payload["chunk_uuid"] == chunk_uuid

    assert point.payload["chunk_id"] == chunks[0]["chunk_id"]

    # Verify original document information.
    assert (
        point.payload["document_name"]
        == document["filename"]
    )

    # Verify text was stored.
    assert point.payload["text"] == chunks[0]["text"]

    # Verify vector exists and has the expected dimension.
    assert point.vector is not None
    assert len(point.vector) == 384

def test_uploaded_document_is_semantically_searchable(
    tmp_path,
    monkeypatch,
):
    """
    Verify that an uploaded document can be found
    through semantic retrieval after being indexed.
    """

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = (
        "Machine learning allows computers to learn "
        "patterns from data without being explicitly programmed."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "machine_learning.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    document_id = data["document"]["document_id"]

    assert data["chunk_count"] > 0

    # Generate a real embedding for a semantic query.
    embedding_service = EmbeddingService()

    query_vector = embedding_service.provider.embed(
        "How do computers learn patterns from data?"
    )

    # Search the same Qdrant collection.
    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    results = store.search(
        query_vector=query_vector,
        top_k=3,
        document_id=document_id,
    )

    assert len(results) > 0

    # The uploaded document must be retrieved.
    assert any(
        result.payload.get("document_id") == document_id
        for result in results
    )

    # The relevant chunk should contain the uploaded content.
    assert any(
        "Machine learning" in result.payload.get("text", "")
        for result in results
    )