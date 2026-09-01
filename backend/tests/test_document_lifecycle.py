from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.storage.document_registry import DocumentRegistry
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.collection_config import LOCAL_COLLECTION_NAME


client = TestClient(app)


def test_complete_document_lifecycle(
    tmp_path,
    monkeypatch,
):
    """
    Verify the complete document lifecycle:

    Upload
        -> Storage
        -> Processing
        -> Chunking
        -> Embedding
        -> Qdrant
        -> Registry

    Delete
        -> Qdrant
        -> Storage
        -> Registry
    """

    # --------------------------------------------------
    # 1. Use temporary local storage
    # --------------------------------------------------

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    # --------------------------------------------------
    # 2. Upload document
    # --------------------------------------------------

    content = (
        "Machine learning is a branch of artificial "
        "intelligence that enables systems to learn "
        "patterns from data."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "lifecycle_test.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    document_id = data["document"]["document_id"]
    storage_path = data["storage_path"]

    assert document_id
    assert data["chunk_count"] > 0

    # --------------------------------------------------
    # 3. Verify physical file exists
    # --------------------------------------------------

    file_path = Path(storage_path)

    assert file_path.exists()

    # --------------------------------------------------
    # 4. Verify document exists in registry
    # --------------------------------------------------

    registry = DocumentRegistry()

    registered_document = registry.get(
        document_id
    )

    assert registered_document is not None

    assert (
        registered_document.document_id
        == document_id
    )

    # --------------------------------------------------
    # 5. Verify vectors exist in Qdrant
    # --------------------------------------------------

    vector_store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    search_results = vector_store.search(
        query_vector=[0.0] * 384,
        top_k=10,
        document_id=document_id,
    )

    assert len(search_results) > 0

    # --------------------------------------------------
    # 6. Delete document
    # --------------------------------------------------

    delete_response = client.delete(
        f"/documents/{document_id}"
    )

    assert delete_response.status_code == 200

    delete_data = delete_response.json()

    assert (
        delete_data["document_id"]
        == document_id
    )

    # --------------------------------------------------
    # 7. Verify physical file is deleted
    # --------------------------------------------------

    assert not file_path.exists()

    # --------------------------------------------------
    # 8. Verify registry entry is deleted
    # --------------------------------------------------

    deleted_document = registry.get(
        document_id
    )

    assert deleted_document is None

    # --------------------------------------------------
    # 9. Verify Qdrant vectors are deleted
    # --------------------------------------------------

    remaining_results = vector_store.search(
        query_vector=[0.0] * 384,
        top_k=10,
        document_id=document_id,
    )

    assert len(remaining_results) == 0

    # --------------------------------------------------
    # 10. Verify API returns 404 after deletion
    # --------------------------------------------------

    get_response = client.get(
        f"/documents/{document_id}"
    )

    assert get_response.status_code == 404