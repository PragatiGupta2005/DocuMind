import uuid
from app.schemas.vector_store_schema import VectorStoreSchema
from app.vector_store.qdrant_store import QdrantVectorStore

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

def test_add_single_embedding(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    record = create_test_record()

    store.add([record])

    assert store.count() >= 1

def test_add_multiple_embeddings(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
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

def test_add_empty_records(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    result = store.add([])

    assert result is None

def test_verify_stored_point(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    record = create_test_record()

    # Store the record
    store.add([record])

    # Retrieve the same point directly from Qdrant
    result = store.client.retrieve(
        collection_name=test_collection,
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

def test_similarity_search(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    test_document_id = f"similarity-test-{uuid.uuid4()}"

    vector_a = [0.0] * 384
    vector_a[0] = 1.0

    vector_b = [0.0] * 384
    vector_b[1] = 1.0

    record_a = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id=test_document_id,
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
        document_id=test_document_id,
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
        document_id=test_document_id,
    )

    assert len(results) == 2

    assert results[0].point_id == record_a.point_id

    assert results[0].score >= results[1].score

    assert (
        results[0].payload["document_id"]
        == test_document_id
    )

def test_search_top_k(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    test_document_id = f"topk-test-{uuid.uuid4()}"

    records = []

    for i in range(5):

        vector = [0.0] * 384
        vector[i] = 1.0

        record = VectorStoreSchema(
            point_id=str(uuid.uuid4()),
            document_id=test_document_id,
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
        document_id=test_document_id,
    )

    assert len(results) == 3
    assert results[0].point_id == records[0].point_id
    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score

def test_search_returns_list(test_collection):

    store = QdrantVectorStore(collection_name=test_collection)
    query_vector = [0.0] * 384
    results = store.search(
        query_vector=query_vector,
        top_k=5,
    )
    assert isinstance(results, list)

def test_search_by_document_id(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    vector_a = [0.0] * 384
    vector_a[0] = 1.0

    vector_b = [0.0] * 384
    vector_b[0] = 0.9
    vector_b[1] = 0.1

    record_a = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="filter-doc-a",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector_a,
        payload={
            "chunk_id": 1,
            "document_name": "filter-a.pdf",
            "text": "Document A content",
        },
    )

    record_b = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="filter-doc-b",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector_b,
        payload={
            "chunk_id": 1,
            "document_name": "filter-b.pdf",
            "text": "Document B content",
        },
    )

    store.add([
        record_a,
        record_b,
    ])

    results = store.search(
        query_vector=vector_a,
        top_k=5,
        document_id="filter-doc-b",
    )

    assert len(results) >= 1

    for result in results:
        assert (
            result.payload["document_id"]
            == "filter-doc-b"
        )

def test_delete_by_document_id(test_collection):

    store = QdrantVectorStore(
        collection_name=test_collection
    )

    vector = [0.0] * 384
    vector[0] = 1.0

    record_a = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="delete-doc-a",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector,
        payload={
            "chunk_id": 1,
            "document_name": "delete-a.pdf",
            "text": "Delete me",
        },
    )

    record_b = VectorStoreSchema(
        point_id=str(uuid.uuid4()),
        document_id="delete-doc-b",
        chunk_uuid=str(uuid.uuid4()),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector,
        payload={
            "chunk_id": 1,
            "document_name": "keep-b.pdf",
            "text": "Keep me",
        },
    )

    store.add([
        record_a,
        record_b,
    ])

    # Delete only document A
    store.delete("delete-doc-a")

    # Document A should no longer exist
    deleted_result = store.client.retrieve(
        collection_name=test_collection,
        ids=[record_a.point_id],
        with_payload=True,
    )

    assert len(deleted_result) == 0

    # Document B should still exist
    remaining_result = store.client.retrieve(
        collection_name=test_collection,
        ids=[record_b.point_id],
        with_payload=True,
    )

    assert len(remaining_result) == 1
    assert (
        remaining_result[0].payload["document_id"]
        == "delete-doc-b"
    )