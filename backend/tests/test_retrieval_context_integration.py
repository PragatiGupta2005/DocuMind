from app.embeddings.embedding_service import EmbeddingService
from app.rag.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalService
from app.vector_store.collection_config import LOCAL_COLLECTION_NAME
from app.vector_store.qdrant_store import QdrantVectorStore


def test_query_retrieval_to_context_integration(
    tmp_path,
    monkeypatch,
):
    """
    Verify the integration:

    Query
        ↓
    RetrievalService
        ↓
    Qdrant semantic search
        ↓
    SearchResultSchema
        ↓
    ContextBuilder
        ↓
    RAGContext
    """

    # Use a temporary directory for uploaded files.
    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    # Import TestClient and app here so the test uses
    # the existing FastAPI application.
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # ---------------------------------------------------------
    # STEP 1: Upload a document
    # ---------------------------------------------------------

    content = (
        "Artificial intelligence enables computers to perform "
        "tasks that normally require human intelligence. "
        "Machine learning is a branch of artificial intelligence "
        "that learns patterns from data."
    )

    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "ai_document.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    upload_data = upload_response.json()

    # Store the complete document information.
    document = upload_data["document"]

    document_id = document["document_id"]

    assert document_id
    assert upload_data["chunk_count"] > 0

    # ---------------------------------------------------------
    # STEP 2: Create the real retrieval service
    # ---------------------------------------------------------

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # ---------------------------------------------------------
    # STEP 3: Perform semantic retrieval
    # ---------------------------------------------------------

    query = "What is machine learning?"

    results = retrieval_service.retrieve(
        query=query,
        top_k=3,
        document_id=document_id,
    )

    assert len(results) > 0

    # The retrieved result must belong to the uploaded document.
    assert all(
        result.payload.get("document_id") == document_id
        for result in results
    )

    # ---------------------------------------------------------
    # STEP 4: Build context from retrieval results
    # ---------------------------------------------------------

    context_builder = ContextBuilder()

    context = context_builder.build(results)

    # ---------------------------------------------------------
    # STEP 5: Verify RAGContext
    # ---------------------------------------------------------

    assert len(context.chunks) == len(results)

    assert context.formatted_context

    # ---------------------------------------------------------
    # STEP 6: Verify retrieved information is preserved
    # ---------------------------------------------------------

    assert document_id in context.formatted_context

    assert "Machine learning" in context.formatted_context

    # The system generates a unique stored filename,
    # so verify the actual filename returned by the API.
    assert document["filename"] in context.formatted_context

    # ---------------------------------------------------------
    # STEP 7: Verify ordering and scores
    # ---------------------------------------------------------

    for result, context_chunk in zip(
        results,
        context.chunks,
    ):
        assert context_chunk.document_id == document_id

        assert (
            context_chunk.chunk_id
            == result.payload["chunk_id"]
        )

        assert (
            context_chunk.text
            == result.payload["text"]
        )

        assert context_chunk.score == result.score