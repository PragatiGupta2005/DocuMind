from app.schemas.document_schema import DocumentSchema
from app.storage.document_registry import DocumentRegistry


def create_document(document_id: str) -> DocumentSchema:
    return DocumentSchema(
        document_id=document_id,
        filename=f"{document_id}.txt",
        file_type="txt",
        text="Test document content.",
        metadata={},
    )


def test_registry_add_and_get(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    document = create_document("doc-001")

    registry.add(document)

    result = registry.get("doc-001")

    assert result is not None
    assert result.document_id == "doc-001"
    assert result.filename == "doc-001.txt"


def test_registry_list(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    registry.add(create_document("doc-001"))
    registry.add(create_document("doc-002"))

    documents = registry.list_all()

    assert len(documents) == 2
    assert documents[0].document_id == "doc-001"
    assert documents[1].document_id == "doc-002"


def test_registry_delete(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    registry.add(create_document("doc-001"))

    deleted = registry.delete("doc-001")

    assert deleted is True
    assert registry.get("doc-001") is None


def test_registry_delete_missing_document(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.storage.document_registry.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    registry = DocumentRegistry()

    deleted = registry.delete("does-not-exist")

    assert deleted is False