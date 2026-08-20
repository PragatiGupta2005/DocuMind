from qdrant_client import QdrantClient

from app.core.vector_store_config import vector_store_settings


class QdrantClientManager:
    """
    Manages the Qdrant client connection.
    """

    def __init__(self):
        self.client = QdrantClient(
            url=vector_store_settings.qdrant_url
        )

    def get_client(self) -> QdrantClient:
        """
        Return the Qdrant client instance.
        """

        return self.client