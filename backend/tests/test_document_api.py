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