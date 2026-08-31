from pathlib import Path

from app.schemas.document_schema import DocumentSchema
from app.services.document_service import DocumentService


def test_delete_document_removes_file_and_registry(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Test document",
        encoding="utf-8",
    )

    service = DocumentService()

    document = DocumentSchema(
        document_id="doc-delete-001",
        filename="test.txt",
        file_type="txt",
        text="Test document",
        metadata={
            "storage_path": str(file_path),
        },
    )

    service.registry.add(document)

    # Replace Qdrant deletion with a fake operation.
    deleted_ids = []

    def fake_delete(document_id):
        deleted_ids.append(document_id)

    service.vector_store.delete = fake_delete

    result = service.delete_document(
        "doc-delete-001"
    )

    assert result is True

    assert not file_path.exists()

    assert (
        service.registry.get(
            "doc-delete-001"
        )
        is None
    )

    assert deleted_ids == [
        "doc-delete-001"
    ]


def test_delete_missing_document_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    service = DocumentService()

    result = service.delete_document(
        "does-not-exist"
    )

    assert result is False