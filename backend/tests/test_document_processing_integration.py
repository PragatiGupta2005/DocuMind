from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_txt_upload_processes_and_chunks_document(
    tmp_path,
    monkeypatch,
):
    """
    Verify that a real TXT upload goes through:

    FastAPI
    → UploadService
    → LocalStorage
    → DocumentProcessingService
    → TXTProcessor
    → ChunkingService
    → UploadResponseSchema
    """

    # Use pytest's temporary directory for uploaded files.
    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = (
        "Machine learning is a branch of artificial intelligence. "
        "It allows computers to learn patterns from data. "
        "Supervised learning uses labeled training data."
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

    # Upload information
    assert data["original_filename"] == "machine_learning.txt"
    assert data["content_type"] == "text/plain"

    # Processed document
    assert data["document"] is not None
    assert data["document"]["text"] == content

    assert data["document"]["document_id"]
    assert data["document"]["filename"]

    # Chunking
    assert data["chunks"] is not None
    assert data["chunk_count"] == len(data["chunks"])
    assert data["chunk_count"] > 0

    # Verify the file was actually stored
    stored_path = Path(data["storage_path"])

    assert stored_path.exists()
    assert stored_path.is_file()


def test_txt_upload_preserves_document_metadata(
    tmp_path,
    monkeypatch,
):
    """
    Verify that document metadata survives the
    upload → processing pipeline.
    """

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = (
        "Artificial intelligence allows machines "
        "to perform tasks that normally require human intelligence."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "ai.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    document = data["document"]

    assert document["text"] == content
    assert document["filename"]
    assert document["file_type"]
    assert document["document_id"]


def test_txt_upload_generates_chunks(
    tmp_path,
    monkeypatch,
):
    """
    Verify that the uploaded document is actually
    passed through the chunking service.
    """

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = (
        "Artificial intelligence is a field of computer science. "
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks to learn representations. "
        "Natural language processing allows computers to process text."
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "ai_topics.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    chunks = data["chunks"]

    assert len(chunks) > 0
    assert data["chunk_count"] == len(chunks)

    # Every returned chunk should belong to the same document.
    document_id = data["document"]["document_id"]

    for chunk in chunks:
        assert chunk["document_id"] == document_id
        assert chunk["text"]
        assert chunk["chunk_uuid"]


def test_upload_creates_unique_document_ids(
    tmp_path,
    monkeypatch,
):
    """
    Verify that separate uploads receive different
    document identifiers.
    """

    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    content = b"Machine learning test document."

    response_1 = client.post(
        "/documents/upload",
        files={
            "file": (
                "document1.txt",
                content,
                "text/plain",
            )
        },
    )

    response_2 = client.post(
        "/documents/upload",
        files={
            "file": (
                "document2.txt",
                content,
                "text/plain",
            )
        },
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    document_1 = response_1.json()["document"]
    document_2 = response_2.json()["document"]

    assert (
        document_1["document_id"]
        != document_2["document_id"]
    )