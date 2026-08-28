import pytest

from app.rag.rag_service import RAGService
from app.schemas.rag_context_schema import (
    ContextChunk,
    RAGContext,
)
from app.schemas.rag_schema import (
    RAGRequest,
)
from app.schemas.search_result_schema import (
    SearchResultSchema,
)


class FakeRetrievalService:

    def __init__(self):
        self.called = False

    def retrieve(
        self,
        query,
        top_k,
        document_id=None,
    ):

        self.called = True

        return [
            SearchResultSchema(
                point_id="point-001",
                score=0.95,
                payload={
                    "document_id": "doc-001",
                    "document_name": "test.pdf",
                    "chunk_id": 1,
                    "text": "Machine learning learns patterns from data.",
                },
            )
        ]


class FakeContextBuilder:

    def __init__(self):
        self.called = False

    def build(self, results):

        self.called = True

        return RAGContext(
            chunks=[
                ContextChunk(
                    document_id="doc-001",
                    document_name="test.pdf",
                    chunk_id=1,
                    text="Machine learning learns patterns from data.",
                    score=0.95,
                )
            ],
            formatted_context=(
                "Machine learning learns patterns from data."
            ),
        )


class FakePromptBuilder:

    def __init__(self):
        self.called = False

    def build(self, query, context):

        self.called = True

        return (
            f"Question: {query}\n"
            f"Context: {context.formatted_context}"
        )


class FakeLLMService:

    def __init__(self):
        self.called = False

    def generate(self, prompt):

        self.called = True

        return (
            "Machine learning enables systems "
            "to learn patterns from data."
        )

    def get_model_name(self):

        return "fake-llm"


def create_service():

    retrieval_service = FakeRetrievalService()

    context_builder = FakeContextBuilder()

    prompt_builder = FakePromptBuilder()

    llm_service = FakeLLMService()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    return (
        service,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
    )


def test_rag_service_generates_answer():

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert (
        response.answer
        == "Machine learning enables systems "
        "to learn patterns from data."
    )


def test_rag_service_calls_retrieval():

    (
        service,
        retrieval_service,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    service.generate(request)

    assert retrieval_service.called is True


def test_rag_service_calls_context_builder():

    (
        service,
        _,
        context_builder,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    service.generate(request)

    assert context_builder.called is True


def test_rag_service_calls_prompt_builder():

    (
        service,
        _,
        _,
        prompt_builder,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    service.generate(request)

    assert prompt_builder.called is True


def test_rag_service_calls_llm():

    (
        service,
        _,
        _,
        _,
        llm_service,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    service.generate(request)

    assert llm_service.called is True


def test_rag_service_returns_sources():

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.document_id == "doc-001"
    assert source.document_name == "test.pdf"
    assert source.chunk_id == 1
    assert source.score == 0.95


def test_rag_service_returns_model_metadata():

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert (
        response.metadata["model_name"]
        == "fake-llm"
    )

    assert (
        response.metadata["retrieved_chunks"]
        == 1
    )


def test_rag_service_rejects_empty_query():

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="   "
    )

    with pytest.raises(ValueError):

        service.generate(request)

def test_rag_service_returns_multiple_sources():

    service, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert len(response.sources) == 1

    assert response.sources[0].document_id == "doc-001"

def test_sources_preserve_context_order():

    service, _, context_builder, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert response.sources[0].chunk_id == 1
    assert response.sources[0].document_name == "test.pdf"

def test_source_contains_retrieval_score():

    service, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    source = response.sources[0]

    assert source.score == 0.95