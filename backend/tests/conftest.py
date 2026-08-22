import pytest

from app.vector_store.qdrant_client import QdrantClientManager
from app.vector_store.qdrant_collection import (
    QdrantCollectionManager,
)
from app.vector_store.collection_config import (
    TEST_LOCAL_COLLECTION_NAME,
    LOCAL_VECTOR_SIZE,
)


@pytest.fixture
def test_collection():

    client = QdrantClientManager().get_client()

    collections = client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if TEST_LOCAL_COLLECTION_NAME in existing_names:
        client.delete_collection(
            collection_name=TEST_LOCAL_COLLECTION_NAME
        )

    manager = QdrantCollectionManager()

    manager.create_collection(
        collection_name=TEST_LOCAL_COLLECTION_NAME,
        vector_size=LOCAL_VECTOR_SIZE,
    )

    yield TEST_LOCAL_COLLECTION_NAME

    client.delete_collection(
        collection_name=TEST_LOCAL_COLLECTION_NAME
    )