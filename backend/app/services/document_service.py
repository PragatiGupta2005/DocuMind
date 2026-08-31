from pathlib import Path

from app.storage.document_registry import DocumentRegistry
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.collection_config import LOCAL_COLLECTION_NAME


class DocumentService:
    """
    Handles document lifecycle operations.
    """

    def __init__(self):
        self.registry = DocumentRegistry()

        self.vector_store = QdrantVectorStore(
            collection_name=LOCAL_COLLECTION_NAME
        )

    def get_document(self, document_id: str):
        """
        Retrieve a document from the registry.
        """

        return self.registry.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from:

        1. Qdrant
        2. Local storage
        3. Document registry

        Returns True when the document was deleted.
        """

        document = self.registry.get(document_id)

        if document is None:
            return False

        # Step 1: Delete vectors from Qdrant
        self.vector_store.delete(document_id)

        # Step 2: Delete physical file
        file_path = Path(
            document.metadata.get(
                "storage_path",
                ""
            )
        )

        if file_path.exists():
            file_path.unlink()

        # Step 3: Delete document metadata
        self.registry.delete(document_id)

        return True