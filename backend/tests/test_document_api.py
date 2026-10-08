from fastapi.testclient import TestClient

from app.main import app
from app.storage.document_registry import DocumentRegistry
from app.schemas.document_schema import DocumentSchema


client = TestClient(app)


def test_get_document_returns_document(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    document = DocumentSchema(
        document_id="doc-001",
        filename="machine-learning.txt",
        file_type="txt",
        text="Machine learning is a branch of AI.",
        metadata={},
    )

    registry.add(document)

    response = client.get(
        "/documents/doc-001"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == "doc-001"
    assert data["filename"] == "machine-learning.txt"
    assert data["file_type"] == "txt"
    assert data["text"] == "Machine learning is a branch of AI."

def test_get_missing_document_returns_404(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )
    # Create an empty registry.
    DocumentRegistry()
    response = client.get(
        "/documents/does-not-exist"
    )
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Document not found"

def test_delete_missing_document_returns_404(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )
    response = client.delete(
        "/documents/does-not-exist"
    )
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Document not found"

def test_list_documents_returns_lightweight_metadata():

    response = client.get("/documents")

    assert response.status_code == 200

    documents = response.json()

    assert isinstance(documents, list)

    if documents:
        document = documents[0]

        assert "document_id" in document
        assert "filename" in document
        assert "file_type" in document

        assert "text" not in document
        assert "metadata" not in document

def test_list_documents_returns_document_metadata(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    document = DocumentSchema(
        document_id="doc-001",
        filename="machine-learning.txt",
        file_type="txt",
        text="Machine learning is a branch of AI.",
        metadata={
            "original_filename": "machine-learning.txt",
            "file_size": 1234,
            "created_at": "2026-10-09T00:00:00+00:00",
            "chunk_count": 5,
        },
    )

    registry.add(document)

    response = client.get("/documents")

    assert response.status_code == 200

    documents = response.json()

    assert len(documents) == 1

    document = documents[0]

    assert document["document_id"] == "doc-001"
    assert document["filename"] == "machine-learning.txt"
    assert document["file_type"] == "txt"

    assert document["original_filename"] == "machine-learning.txt"
    assert document["file_size"] == 1234
    assert document["created_at"] == "2026-10-09T00:00:00+00:00"
    assert document["chunk_count"] == 5

    # Listing endpoint must remain lightweight.
    assert "text" not in document
    assert "metadata" not in document