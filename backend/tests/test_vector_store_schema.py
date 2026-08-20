import pytest

from app.schemas.vector_store_schema import VectorStoreSchema


def create_vector(
    dimensions: int
) -> list[float]:
    """
    Create a test vector with the required number
    of dimensions.
    """

    return [0.1] * dimensions


def test_valid_384_dimension_vector():
    """
    Test a valid local embedding vector.
    """

    vector = create_vector(384)

    record = VectorStoreSchema(
        point_id="chunk-001",
        document_id="doc-001",
        chunk_uuid="chunk-uuid-001",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector,
        payload={
            "chunk_id": 1,
            "document_name": "AI.pdf",
            "text": "Artificial Intelligence is amazing."
        }
    )

    assert record.point_id == "chunk-001"
    assert record.document_id == "doc-001"
    assert record.chunk_uuid == "chunk-uuid-001"

    assert record.model_name == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert record.dimensions == 384
    assert len(record.vector) == 384

    assert record.payload["chunk_id"] == 1
    assert record.payload["document_name"] == "AI.pdf"


def test_valid_3072_dimension_vector():
    """
    Test a valid Gemini embedding vector.
    """

    vector = create_vector(3072)

    record = VectorStoreSchema(
        point_id="chunk-002",
        document_id="doc-001",
        chunk_uuid="chunk-uuid-002",
        model_name="gemini-embedding-001",
        dimensions=3072,
        vector=vector,
        payload={
            "chunk_id": 2,
            "document_name": "AI.pdf",
            "text": "Machine Learning is a subset of AI."
        }
    )

    assert record.model_name == "gemini-embedding-001"
    assert record.dimensions == 3072
    assert len(record.vector) == 3072


def test_dimension_mismatch():
    """
    Test that an incorrect vector dimension
    is rejected.
    """

    vector = create_vector(384)

    with pytest.raises(
        ValueError,
        match="Vector dimensions do not match"
    ):
        VectorStoreSchema(
            point_id="chunk-003",
            document_id="doc-001",
            chunk_uuid="chunk-uuid-003",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=3072,
            vector=vector,
            payload={}
        )


def test_payload_storage():
    """
    Test that payload data is preserved.
    """

    vector = create_vector(384)

    payload = {
        "chunk_id": 5,
        "document_name": "AI.pdf",
        "text": "Deep Learning uses Neural Networks.",
        "start_index": 200,
        "end_index": 245
    }

    record = VectorStoreSchema(
        point_id="chunk-005",
        document_id="doc-001",
        chunk_uuid="chunk-uuid-005",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        vector=vector,
        payload=payload
    )

    assert record.payload["chunk_id"] == 5
    assert record.payload["document_name"] == "AI.pdf"
    assert record.payload["text"] == (
        "Deep Learning uses Neural Networks."
    )
    assert record.payload["start_index"] == 200
    assert record.payload["end_index"] == 245


def test_required_fields():
    """
    Test that required fields cannot be omitted.
    """

    with pytest.raises(ValueError):

        VectorStoreSchema(
            point_id="chunk-006",
            document_id="doc-001",
            chunk_uuid="chunk-uuid-006",
            model_name="test-model",
            dimensions=384,
            # vector intentionally omitted
            payload={}
        )