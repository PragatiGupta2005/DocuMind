from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class FakeLLMService:
    """
    Deterministic LLM service used for end-to-end testing.

    This avoids calling the real Gemini API while still
    verifying that the RAG pipeline reaches the LLM stage.
    """

    def __init__(self):
        self.received_prompts = []

    def generate(self, prompt: str) -> str:
        self.received_prompts.append(prompt)

        return (
            "Machine learning allows computers to learn "
            "patterns from data."
        )

    def get_model_name(self) -> str:
        return "fake-test-llm"


def test_upload_index_query_end_to_end(
    tmp_path,
    monkeypatch,
):
    """
    Verify the complete RAG pipeline:

    Upload
        ↓
    Process
        ↓
    Chunk
        ↓
    Embed
        ↓
    Index in Qdrant
        ↓
    RAG Query
        ↓
    Retrieval
        ↓
    Context
        ↓
    Prompt
        ↓
    Fake LLM
        ↓
    RAG Response
    """

    # Use a temporary directory for uploaded files.
    monkeypatch.setattr(
        "app.storage.local_storage.UPLOAD_DIRECTORY",
        str(tmp_path),
    )

    # Replace the real LLM service used by the RAG API.
    fake_llm_service = FakeLLMService()

    monkeypatch.setattr(
        "app.api.rag.LLMService",
        lambda: fake_llm_service,
    )

    # ---------------------------------------------------------
    # STEP 1: Upload a real document
    # ---------------------------------------------------------

    content = (
        "Machine learning is a branch of artificial intelligence. "
        "Machine learning systems learn patterns from data. "
        "These systems can use learned patterns to make predictions."
    )

    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "machine_learning_rag.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    upload_data = upload_response.json()

    document = upload_data["document"]

    document_id = document["document_id"]

    # ---------------------------------------------------------
    # STEP 2: Verify document was processed and indexed
    # ---------------------------------------------------------

    assert document_id
    assert upload_data["chunk_count"] > 0
    assert len(upload_data["chunks"]) == upload_data["chunk_count"]

    # ---------------------------------------------------------
    # STEP 3: Send a real RAG query
    # ---------------------------------------------------------

    query_response = client.post(
        "/rag/query",
        json={
            "query": "How do machine learning systems learn?",
            "top_k": 3,
            "document_id": document_id,
        },
    )

    assert query_response.status_code == 200

    rag_data = query_response.json()

    # ---------------------------------------------------------
    # STEP 4: Verify RAG answer
    # ---------------------------------------------------------

    assert rag_data["answer"]

    assert (
        rag_data["answer"]
        == "Machine learning allows computers to learn "
        "patterns from data."
    )

    # ---------------------------------------------------------
    # STEP 5: Verify retrieved sources
    # ---------------------------------------------------------

    assert len(rag_data["sources"]) > 0

    assert any(
        source["document_id"] == document_id
        for source in rag_data["sources"]
    )

    # ---------------------------------------------------------
    # STEP 6: Verify source metadata
    # ---------------------------------------------------------

    matching_sources = [
        source
        for source in rag_data["sources"]
        if source["document_id"] == document_id
    ]

    assert len(matching_sources) > 0

    for source in matching_sources:
        assert source["document_name"] == document["filename"]
        assert source["chunk_id"] >= 0
        assert source["score"] >= 0

    # ---------------------------------------------------------
    # STEP 7: Verify the prompt reached the LLM
    # ---------------------------------------------------------

    assert len(fake_llm_service.received_prompts) == 1

    generated_prompt = fake_llm_service.received_prompts[0]

    assert "How do machine learning systems learn?" in generated_prompt
    assert "Machine learning" in generated_prompt

    # ---------------------------------------------------------
    # STEP 8: Verify LLM metadata
    # ---------------------------------------------------------

    assert rag_data["metadata"]["model_name"] == "fake-test-llm"

    assert (
        rag_data["metadata"]["retrieved_chunks"]
        == len(rag_data["sources"])
    )