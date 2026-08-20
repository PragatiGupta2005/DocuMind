from app.vector_store.qdrant_collection import (
    QdrantCollectionManager,
)
from app.vector_store.collection_config import (
    LOCAL_COLLECTION_NAME,
    API_COLLECTION_NAME,
)


def test_qdrant_connection():

    manager = QdrantCollectionManager()

    manager.client.get_collections()

    assert manager.client is not None


def test_qdrant_collections():

    manager = QdrantCollectionManager()

    manager.setup_collections()

    collections = (
        manager.client.get_collections()
    )

    collection_names = {
        collection.name
        for collection in collections.collections
    }

    assert LOCAL_COLLECTION_NAME in collection_names

    assert API_COLLECTION_NAME in collection_names