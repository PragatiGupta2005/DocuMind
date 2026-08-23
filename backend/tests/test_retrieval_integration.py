from app.embeddings.embedding_service import EmbeddingService
from app.schemas.chunk_schema import ChunkSchema
from app.services.retrieval_service import RetrievalService
from app.vector_store.qdrant_store import QdrantVectorStore
import uuid

def test_real_local_embedding_qdrant_retrieval(
    test_collection,
):
    """
    Verify the complete retrieval pipeline using
    the real local embedding provider and Qdrant.
    """

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore(
        collection_name=test_collection
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chunks = [
        ChunkSchema(
            document_id="retrieval-doc-001",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=1,
            document_name="machine-learning.pdf",
            text=(
                "Machine learning is a branch of artificial "
                "intelligence that enables systems to learn "
                "patterns from data."
            ),
            start_index=0,
            end_index=130,
            metadata={},
        ),
        ChunkSchema(
            document_id="retrieval-doc-001",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=2,
            document_name="machine-learning.pdf",
            text=(
                "Supervised learning uses labeled training data "
                "to learn a mapping between inputs and outputs."
            ),
            start_index=131,
            end_index=260,
            metadata={},
        ),
        ChunkSchema(
            document_id="retrieval-doc-002",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=1,
            document_name="databases.pdf",
            text=(
                "A database is an organized collection of data "
                "that can be stored, managed, and retrieved."
            ),
            start_index=0,
            end_index=110,
            metadata={},
        ),
    ]

    embeddings = embedding_service.embed_chunks(
        chunks
    )

    assert len(embeddings) == 3

    assert all(
        len(embedding.vector) == 384
        for embedding in embeddings
    )

    # Convert embeddings into VectorStoreSchema records.
    from app.schemas.vector_store_schema import (
        VectorStoreSchema,
    )

    records = []

    for chunk, embedding in zip(chunks, embeddings):

        records.append(
            VectorStoreSchema(
                point_id=chunk.chunk_uuid,
                document_id=embedding.document_id,
                chunk_uuid=embedding.chunk_uuid,
                model_name=embedding.model_name,
                dimensions=embedding.dimensions,
                vector=embedding.vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "text": chunk.text,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                },
            )
        )

    vector_store.add(records)

    results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=3,
    )

    assert len(results) > 0

    assert results[0].payload is not None

    assert "text" in results[0].payload

    assert "document_id" in results[0].payload