from qdrant_client.models import Distance, VectorParams

from app.vector_store.collection_config import (
    LOCAL_COLLECTION_NAME,
    LOCAL_VECTOR_SIZE,
    API_COLLECTION_NAME,
    API_VECTOR_SIZE,
)
from app.vector_store.qdrant_client import (
    QdrantClientManager,
)


class QdrantCollectionManager:
    """
    Handles creation and management of Qdrant collections.
    """

    def __init__(self):
        self.client = (
            QdrantClientManager()
            .get_client()
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ):
        """
        Create a collection if it does not already exist.
        """

        collections = (
            self.client.get_collections()
        )

        existing_names = {
            collection.name
            for collection in collections.collections
        }

        if collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def setup_collections(self):
        """
        Create all DocuMind embedding collections.
        """

        self.create_collection(
            collection_name=LOCAL_COLLECTION_NAME,
            vector_size=LOCAL_VECTOR_SIZE,
        )

        self.create_collection(
            collection_name=API_COLLECTION_NAME,
            vector_size=API_VECTOR_SIZE,
        )