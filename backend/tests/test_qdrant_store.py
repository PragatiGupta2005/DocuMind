import uuid

from app.schemas.vector_store_schema import VectorStoreSchema
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.collection_config import (
    LOCAL_COLLECTION_NAME,
)


def create_test_record():

    vector = [0.1] * 384

    return VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="test-doc-001",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector,
        payload={
            "chunk_id": 1,
            "document_name": "test.pdf",
            "text": "Artificial Intelligence is amazing.",
            "start_index": 0,
            "end_index": 38,
        },
    )


def test_add_single_embedding():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    record = create_test_record()

    store.add([record])

    assert store.count() >= 1


def test_add_multiple_embeddings():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    records = []

    for i in range(3):

        vector = [0.1 + (i * 0.001)] * 384

        record = VectorStoreSchema(
            point_id=str(uuid.uuid4()),
            document_id="test-doc-batch",
            chunk_uuid=str(uuid.uuid4()),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            vector=vector,
            payload={
                "chunk_id": i + 1,
                "document_name": "batch-test.pdf",
                "text": f"Test chunk {i + 1}",
            },
        )

        records.append(record)

    store.add(records)

    assert store.count() >= 3


def test_add_empty_records():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    result = store.add([])

    assert result is None

def test_verify_stored_point():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    record = create_test_record()

    # Store the record
    store.add([record])

    # Retrieve the same point directly from Qdrant
    result = store.client.retrieve(
        collection_name=LOCAL_COLLECTION_NAME,
        ids=[record.point_id],
        with_payload=True,
        with_vectors=True,
    )

    assert len(result) == 1

    point = result[0]

    # Verify point ID
    assert str(point.id) == record.point_id

    # Verify vector
    assert point.vector is not None
    assert len(point.vector) == record.dimensions

    # Qdrant normalizes vectors when using cosine distance.
    magnitude = sum(
        value ** 2
        for value in point.vector
    ) ** 0.5

    assert abs(magnitude - 1.0) < 1e-6

    # Verify payload
    assert point.payload["document_id"] == record.document_id
    assert point.payload["chunk_uuid"] == record.chunk_uuid
    assert point.payload["model_name"] == record.model_name
    assert point.payload["dimensions"] == record.dimensions
    assert point.payload["chunk_id"] == 1
    assert point.payload["document_name"] == "test.pdf"
    assert point.payload["text"] == ("Artificial Intelligence is amazing.")
    assert point.payload["start_index"] == 0
    assert point.payload["end_index"] == 38

def test_similarity_search():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    vector_a = [0.0] * 384
    vector_a[0] = 1.0

    vector_b = [0.0] * 384
    vector_b[1] = 1.0

    record_a = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="search-doc-a",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector_a,
        payload={
            "chunk_id": 1,
            "document_name": "search-a.pdf",
            "text": "Artificial Intelligence",
        },
    )

    record_b = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="search-doc-b",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector_b,
        payload={
            "chunk_id": 2,
            "document_name": "search-b.pdf",
            "text": "Database Systems",
        },
    )

    store.add([
        record_a,
        record_b,
    ])

    results = store.search(
        query_vector=vector_a,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].point_id == record_a.point_id
    assert results[0].score >= results[1].score
    assert results[0].payload["document_id"] == ("search-doc-a")

def test_search_top_k():

    store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    records = []

    for i in range(5):

        vector = [0.0] * 384
        vector[i] = 1.0

        record = VectorStoreSchema(
            point_id=str(uuid.uuid4()),
            document_id=f"topk-doc-{i}",
            chunk_uuid=str(uuid.uuid4()),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            vector=vector,
            payload={
                "chunk_id": i + 1,
                "document_name": f"topk-{i}.pdf",
                "text": f"Top K test chunk {i}",
            },
        )

        records.append(record)

    store.add(records)

    query_vector = [0.0] * 384
    query_vector[0] = 1.0

    results = store.search(
        query_vector=query_vector,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].point_id == records[0].point_id

def test_search_returns_list():

    store = QdrantVectorStore(collection_name=LOCAL_COLLECTION_NAME)
    query_vector = [0.0] * 384
    results = store.search(
        query_vector=query_vector,
        top_k=5,
    )
    assert isinstance(results, list)