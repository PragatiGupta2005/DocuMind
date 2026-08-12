from app.schemas.embedding_schema import EmbeddingSchema


embedding = EmbeddingSchema(
    document_id="doc_001",
    chunk_uuid="chunk_001",
    model_name="test-model",
    dimensions=4,
    vector=[
        0.12,
        -0.45,
        0.78,
        0.21
    ],
    metadata={
        "page": 1
    }
)

print(embedding.model_dump_json(indent=4))