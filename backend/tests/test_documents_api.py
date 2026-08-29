from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_document():

    mock_response = {
        "filename": "test-unique.pdf",
        "original_filename": "test.pdf",
        "size": 100,
        "content_type": "application/pdf",
        "storage_path": "uploads/test-unique.pdf",
        "message": "File uploaded, processed, and chunked successfully",
        "document": {
            "document_id": "doc-001",
            "filename": "test-unique.pdf",
            "file_type": "pdf",
            "text": "Test document content",
            "metadata": {},
        },
        "chunks": [],
        "chunk_count": 0,
    }

    with patch(
        "app.api.documents.UploadService"
    ) as mock_service:

        mock_service.return_value.upload_file = (
            AsyncMock(
                return_value=mock_response
            )
        )

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    b"Test document content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["original_filename"] == "test.pdf"
    assert data["chunk_count"] == 0