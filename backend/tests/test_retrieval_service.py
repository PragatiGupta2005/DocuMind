from app.core.settings import MIN_RELEVANCE_SCORE
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

    def __init__(self, results=None):
        self.received_vector = None
        self.received_top_k = None
        self.received_document_id = None

        # Allow individual tests to control search results
        self.results = results

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ):

        self.received_vector = query_vector
        self.received_top_k = top_k
        self.received_document_id = document_id

        if self.results is not None:
            return self.results

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


def create_search_result(
    point_id: str,
    score: float,
) -> SearchResultSchema:

    return SearchResultSchema(
        point_id=point_id,
        score=score,
        payload={
            "document_id": "doc-001",
            "text": "Test document content",
        },
    )


# ============================================================
# Basic RetrievalService Tests
# ============================================================

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


# ============================================================
# 8.3.3 — Relevance Threshold Tests
# ============================================================

def test_retrieval_keeps_results_above_threshold():

    results = [
        create_search_result(
            point_id="point-high",
            score=0.90,
        ),
        create_search_result(
            point_id="point-medium",
            score=0.70,
        ),
        create_search_result(
            point_id="point-low",
            score=0.40,
        ),
    ]

    embedding_service = FakeEmbeddingService()

    vector_store = FakeVectorStore(
        results=results
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    filtered_results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    assert len(filtered_results) == 2

    assert filtered_results[0].score == 0.90
    assert filtered_results[1].score == 0.70

    assert all(
        result.score >= MIN_RELEVANCE_SCORE
        for result in filtered_results
    )


def test_retrieval_keeps_result_equal_to_threshold():

    result = create_search_result(
        point_id="point-boundary",
        score=MIN_RELEVANCE_SCORE,
    )

    embedding_service = FakeEmbeddingService()

    vector_store = FakeVectorStore(
        results=[result]
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    filtered_results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    assert len(filtered_results) == 1

    assert (
        filtered_results[0].score
        == MIN_RELEVANCE_SCORE
    )


def test_retrieval_removes_results_below_threshold():

    results = [
        create_search_result(
            point_id="point-low-1",
            score=MIN_RELEVANCE_SCORE - 0.01,
        ),
        create_search_result(
            point_id="point-low-2",
            score=0.30,
        ),
        create_search_result(
            point_id="point-low-3",
            score=0.10,
        ),
    ]

    embedding_service = FakeEmbeddingService()

    vector_store = FakeVectorStore(
        results=results
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    filtered_results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    assert filtered_results == []


def test_retrieval_threshold_can_reduce_final_result_count():

    results = [
        create_search_result(
            point_id="point-1",
            score=0.90,
        ),
        create_search_result(
            point_id="point-2",
            score=0.80,
        ),
        create_search_result(
            point_id="point-3",
            score=0.60,
        ),
        create_search_result(
            point_id="point-4",
            score=0.40,
        ),
        create_search_result(
            point_id="point-5",
            score=0.20,
        ),
    ]

    embedding_service = FakeEmbeddingService()

    vector_store = FakeVectorStore(
        results=results
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    filtered_results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    # Vector store returned 5 candidates
    assert vector_store.received_top_k == 5

    # Only 3 satisfy the configured threshold
    assert len(filtered_results) == 3

    assert all(
        result.score >= MIN_RELEVANCE_SCORE
        for result in filtered_results
    )


def test_retrieval_returns_no_results_when_all_are_below_threshold():

    results = [
        create_search_result(
            point_id="point-1",
            score=MIN_RELEVANCE_SCORE - 0.01,
        ),
        create_search_result(
            point_id="point-2",
            score=0.30,
        ),
        create_search_result(
            point_id="point-3",
            score=0.10,
        ),
    ]

    embedding_service = FakeEmbeddingService()

    vector_store = FakeVectorStore(
        results=results
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    filtered_results = retrieval_service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    assert filtered_results == []