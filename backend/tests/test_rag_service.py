import pytest
from app.rag.rag_service import RAGService
from app.schemas.rag_context_schema import (ContextChunk,RAGContext)
from app.schemas.rag_schema import (RAGRequest,)
from app.schemas.search_result_schema import (
    SearchResultSchema,
)


# ============================================================
# Fake Retrieval Service
# ============================================================

class FakeRetrievalService:

    def __init__(self, results=None):

        self.called = False
        self.received_query = None
        self.received_top_k = None
        self.received_document_id = None

        self.results = results

    def retrieve(
        self,
        query,
        top_k,
        document_id=None,
    ):

        self.called = True

        self.received_query = query
        self.received_top_k = top_k
        self.received_document_id = document_id

        # Allow individual tests to control retrieval results
        if self.results is not None:
            return self.results

        # Default fake retrieval result
        return [
            SearchResultSchema(
                point_id="point-001",
                score=0.95,
                payload={
                    "document_id": "doc-001",
                    "document_name": "test.pdf",
                    "chunk_id": 1,
                    "text": (
                        "Machine learning learns patterns "
                        "from data."
                    ),
                },
            )
        ]


# ============================================================
# Fake Context Builder
# ============================================================

class FakeContextBuilder:

    def __init__(self):

        self.called = False
        self.received_results = None

    def build(self, results):

        self.called = True
        self.received_results = results

        # No retrieval results
        if not results:

            return RAGContext(
                chunks=[],
                formatted_context="",
            )

        # Normal retrieval result
        return RAGContext(
            chunks=[
                ContextChunk(
                    document_id="doc-001",
                    document_name="test.pdf",
                    chunk_id=1,
                    text=(
                        "Machine learning enables systems "
                        "to learn patterns from data."
                    ),
                    score=0.95,
                )
            ],
            formatted_context=(
                "Machine learning enables systems "
                "to learn patterns from data."
            ),
        )


# ============================================================
# Fake Prompt Builder
# ============================================================

class FakePromptBuilder:

    def __init__(self):

        self.called = False
        self.received_query = None
        self.received_context = None

    def build(
        self,
        query,
        context,
    ):

        self.called = True

        self.received_query = query
        self.received_context = context

        return (
            f"Question: {query}\n"
            f"Context: {context.formatted_context}"
        )


# ============================================================
# Fake LLM Service
# ============================================================

class FakeLLMService:
    def __init__(self):
        self.called = False
        self.received_prompt = None

    def generate(self, prompt):
        self.called = True
        self.received_prompt = prompt
        return (
            "Machine learning enables systems "
            "to learn patterns from data."
        )

    def get_model_name(self):
        return "fake-llm"
    

# ============================================================
# Service Factory
# ============================================================

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


# ============================================================
# Tests
# ============================================================

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


def test_rag_request_rejects_empty_query():

    with pytest.raises(ValueError):

        RAGRequest(
            query="   "
        )


def test_rag_service_returns_multiple_sources():

    service, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert len(response.sources) == 1

    assert (
        response.sources[0].document_id
        == "doc-001"
    )


def test_sources_preserve_context_order():

    service, _, context_builder, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert response.sources[0].chunk_id == 1
    assert (
        response.sources[0].document_name
        == "test.pdf"
    )


def test_source_contains_retrieval_score():

    service, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    source = response.sources[0]

    assert source.score == 0.95


def test_rag_service_passes_retrieval_parameters():

    (
        service,
        retrieval_service,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?",
        top_k=3,
        document_id="doc-123",
    )

    service.generate(request)

    assert retrieval_service.called is True

    assert (
        retrieval_service.received_query
        == "What is machine learning?"
    )

    assert (
        retrieval_service.received_top_k
        == 3
    )

    assert (
        retrieval_service.received_document_id
        == "doc-123"
    )


# ============================================================
# Phase 7.2.3
# No-result retrieval path
# ============================================================

def test_rag_service_handles_no_retrieval_results():

    retrieval_service = FakeRetrievalService(
        results=[]
    )

    context_builder = FakeContextBuilder()

    prompt_builder = FakePromptBuilder()

    llm_service = FakeLLMService()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    request = RAGRequest(
        query="What is quantum computing?"
    )

    response = service.generate(request)

    # --------------------------------------------------------
    # 1. Retrieval was called
    # --------------------------------------------------------

    assert retrieval_service.called is True

    # --------------------------------------------------------
    # 2. Retrieval actually returned no results
    # --------------------------------------------------------

    assert (
        context_builder.received_results
        == []
    )

    # --------------------------------------------------------
    # 3. Context builder produced empty context
    # --------------------------------------------------------

    assert context_builder.called is True

    assert (
        prompt_builder.received_context.chunks
        == []
    )

    assert (
        prompt_builder.received_context.formatted_context
        == ""
    )

    # --------------------------------------------------------
    # 4. LLM was still called
    # --------------------------------------------------------

    assert llm_service.called is True

    # --------------------------------------------------------
    # 5. RAG still returns an answer
    # --------------------------------------------------------

    assert (
        response.answer
        == "Machine learning enables systems "
        "to learn patterns from data."
    )

    # --------------------------------------------------------
    # 6. There should be no sources
    # --------------------------------------------------------

    assert response.sources == []

    # --------------------------------------------------------
    # 7. Metadata should report zero retrieved chunks
    # --------------------------------------------------------

    assert (
        response.metadata["retrieved_chunks"]
        == 0
    )

def test_rag_service_passes_generated_prompt_to_llm():
    service, _, _, prompt_builder, llm_service = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert llm_service.called is True
    assert llm_service.received_prompt is not None
    assert "What is machine learning?" in llm_service.received_prompt
    assert (
        response.answer
        == "Machine learning enables systems "
        "to learn patterns from data."
    )