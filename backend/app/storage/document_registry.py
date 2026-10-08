import json
from pathlib import Path

from app.constants.file_constants import UPLOAD_DIRECTORY
from app.schemas.document_schema import DocumentSchema


class DocumentRegistry:
    """
    Stores lightweight metadata about uploaded documents.
    """

    def __init__(self):
        self.registry_path = (
            Path(UPLOAD_DIRECTORY) / "documents.json"
        )

        if not self.registry_path.exists():
            self.registry_path.write_text(
                "[]",
                encoding="utf-8",
            )

    def _read(self) -> list[dict]:
        return json.loads(
            self.registry_path.read_text(
                encoding="utf-8"
            )
        )

    def _write(self, documents: list[dict]) -> None:
        self.registry_path.write_text(
            json.dumps(
                documents,
                indent=2,
            ),
            encoding="utf-8",
        )

    def add(
        self,
        document: DocumentSchema,
    ) -> None:

        documents = self._read()

        documents.append(
            document.model_dump()
        )

        self._write(documents)

    def get(
        self,
        document_id: str,
    ) -> DocumentSchema | None:

        documents = self._read()

        for document in documents:
            if document["document_id"] == document_id:
                return DocumentSchema(**document)

        return None

    def list_all(self) -> list[DocumentSchema]:

        documents = self._read()

        return [
            DocumentSchema(**document)
            for document in documents
        ]

    def delete(
        self,
        document_id: str,
    ) -> bool:

        documents = self._read()

        remaining = [
            document
            for document in documents
            if document["document_id"] != document_id
        ]

        deleted = len(remaining) != len(documents)

        if deleted:
            self._write(remaining)

        return deleted

    def get_statistics(self) -> dict:
        documents = self._read()

        total_documents = len(documents)
        total_chunks = 0
        total_size = 0
        file_types = {}

        for document in documents:
            metadata = document.get("metadata", {})

            total_chunks += metadata.get("chunk_count", 0)
            total_size += metadata.get("file_size", 0)

            file_type = document.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_size": total_size,
            "file_types": file_types,
        }