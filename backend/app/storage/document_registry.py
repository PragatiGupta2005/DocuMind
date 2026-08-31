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