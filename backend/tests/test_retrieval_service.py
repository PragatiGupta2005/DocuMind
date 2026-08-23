from app.schemas.search_result_schema import SearchResultSchema
from app.services.retrieval_service import RetrievalService

class FakeEmbeddingService:
    """
    Fake embedding service used for unit testing.
    """

    def __init__(self):
        self.received_text = None

    def embed_text(self, text: str) -> list[float]:

        self.received_text = text

        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    """
    Fake vector store used for unit testing.
    """

    def __init__(self):
        self.received_vector = None
        self.received_top_k = None
        self.received_document_id = None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ):

        self.received_vector = query_vector
        self.received_top_k = top_k
        self.received_document_id = document_id

        return [
            SearchResultSchema(
                point_id="test-point-1",
                score=0.95,
                payload={
                    "document_id": "doc-001",
                    "text": "Test document content",
                },
            )
        ]


def test_retrieval_service_generates_query_embedding():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service.retrieve(
        query="What is machine learning?"
    )

    assert (
        embedding_service.received_text
        == "What is machine learning?"
    )

def test_retrieval_service_passes_embedding_to_vector_store():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service.retrieve(
        query="What is machine learning?"
    )

    assert vector_store.received_vector == [
        0.1,
        0.2,
        0.3,
    ]

def test_retrieval_service_passes_top_k():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=3,
    )

    assert vector_store.received_top_k == 3

def test_retrieval_service_passes_document_id():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
        document_id="doc-001",
    )

    assert (
        vector_store.received_document_id
        == "doc-001"
    )

def test_retrieval_service_rejects_empty_query():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    try:
        retrieval_service.retrieve("")

        assert False, "Expected ValueError"

    except ValueError as error:

        assert str(error) == "Query cannot be empty."

def test_retrieval_service_returns_search_results():

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retrieval_service.retrieve(
        query="What is machine learning?"
    )

    assert len(results) == 1

    assert results[0].point_id == "test-point-1"

    assert results[0].score == 0.95

    assert (
        results[0].payload["document_id"]
        == "doc-001"
    )