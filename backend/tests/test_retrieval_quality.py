import uuid

from app.embeddings.embedding_service import EmbeddingService
from app.schemas.chunk_schema import ChunkSchema
from app.schemas.vector_store_schema import VectorStoreSchema
from app.services.retrieval_service import RetrievalService
from app.vector_store.qdrant_store import QdrantVectorStore


def create_chunks():

    return [
        ChunkSchema(
            document_id="quality-doc-001",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=1,
            document_name="ai-guide.pdf",
            text=(
                "Machine learning is a branch of artificial "
                "intelligence that allows computers to learn "
                "patterns from data and make predictions."
            ),
            start_index=0,
            end_index=130,
            metadata={},
        ),

        ChunkSchema(
            document_id="quality-doc-001",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=2,
            document_name="ai-guide.pdf",
            text=(
                "Supervised learning trains a model using labeled "
                "examples where the desired output is already known."
            ),
            start_index=131,
            end_index=270,
            metadata={},
        ),

        ChunkSchema(
            document_id="quality-doc-001",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=3,
            document_name="ai-guide.pdf",
            text=(
                "Neural networks consist of interconnected layers "
                "of artificial neurons and are widely used for "
                "complex pattern recognition."
            ),
            start_index=271,
            end_index=410,
            metadata={},
        ),

        ChunkSchema(
            document_id="quality-doc-002",
            chunk_uuid=str(uuid.uuid4()),
            chunk_id=1,
            document_name="database-guide.pdf",
            text=(
                "A database is an organized collection of data "
                "that allows information to be stored, managed, "
                "and retrieved efficiently."
            ),
            start_index=0,
            end_index=120,
            metadata={},
        ),
    ]

def setup_retrieval_system(
    test_collection,
):

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore(
        collection_name=test_collection
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chunks = create_chunks()

    embeddings = embedding_service.embed_chunks(
        chunks
    )

    records = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

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

    return retrieval_service

def test_retrieval_returns_relevant_chunk(
    test_collection,
):

    retrieval_service = setup_retrieval_system(
        test_collection
    )

    results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=3,
    )

    assert len(results) == 3

    texts = [
        result.payload["text"]
        for result in results
    ]

    assert any(
        "Machine learning" in text
        for text in texts
    )

def test_relevant_chunk_has_highest_score(
    test_collection,
):

    retrieval_service = setup_retrieval_system(
        test_collection
    )

    results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=4,
    )

    assert len(results) == 4

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

def test_retrieval_respects_top_k(
    test_collection,
):

    retrieval_service = setup_retrieval_system(
        test_collection
    )

    results = retrieval_service.retrieve(
        query="What is artificial intelligence?",
        top_k=2,
    )

    assert len(results) == 2

def test_retrieval_document_filter(
    test_collection,
):

    retrieval_service = setup_retrieval_system(
        test_collection
    )

    results = retrieval_service.retrieve(
        query="How is information stored and retrieved?",
        top_k=5,
        document_id="quality-doc-002",
    )

    assert len(results) >= 1

    for result in results:

        assert (
            result.payload["document_name"]
            == "database-guide.pdf"
        )

def test_semantic_retrieval_with_paraphrased_query(
    test_collection,
):

    retrieval_service = setup_retrieval_system(
        test_collection
    )

    results = retrieval_service.retrieve(
        query=(
            "How can computers learn patterns "
            "from examples and data?"
        ),
        top_k=3,
    )

    assert len(results) == 3

    texts = [
        result.payload["text"]
        for result in results
    ]

    assert any(
        "Machine learning" in text
        for text in texts
    )

    