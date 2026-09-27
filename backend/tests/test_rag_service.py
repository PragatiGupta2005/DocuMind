import pytest
import logging
from app.rag.rag_service import RAGService
from app.schemas.rag_context_schema import (
    ContextChunk,
    RAGContext,
)
from app.schemas.rag_schema import RAGRequest
from app.schemas.search_result_schema import SearchResultSchema
from app.schemas.answer_validation_schema import (
    AnswerValidationResult,
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

        if self.results is not None:
            return self.results

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

        if not results:
            return RAGContext(
                chunks=[],
                formatted_context="",
            )

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
# Fake Answer Validator
# ============================================================

class FakeAnswerValidator:

    def __init__(self):
        self.called = False
        self.received_answer = None
        self.received_context = None

    def validate(self, answer, context):

        self.called = True
        self.received_answer = answer
        self.received_context = context

        return AnswerValidationResult(
            is_grounded=True,
            reason="Test answer is grounded.",
        )


# ============================================================
# Fake Source Validator
# ============================================================

class FakeSourceValidator:

    def __init__(self):
        self.called = False
        self.received_context = None
        self.received_sources = None

    def validate(self, context, sources):

        self.called = True
        self.received_context = context
        self.received_sources = sources

        return True


# ============================================================
# Service Factory
# ============================================================

def create_service():

    retrieval_service = FakeRetrievalService()

    context_builder = FakeContextBuilder()

    prompt_builder = FakePromptBuilder()

    llm_service = FakeLLMService()

    answer_validator = FakeAnswerValidator()

    source_validator = FakeSourceValidator()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        answer_validator=answer_validator,
        source_validator=source_validator,
    )

    return (
        service,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
        answer_validator,
        source_validator,
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

def test_rag_service_logs_request(caplog):
    service, _, _, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    with caplog.at_level(logging.INFO):
        service.generate(request)

    assert "RAG request received" in caplog.text

def test_rag_service_logs_retrieval_metadata(caplog):

    service, _, _, _, _, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?",
        top_k=5,
    )

    with caplog.at_level(logging.INFO):
        service.generate(request)

    assert "Retrieval completed" in caplog.text
    assert "retrieved_chunks=1" in caplog.text
    assert "top_k=5" in caplog.text
    assert "document_id=None" in caplog.text
    
def test_rag_service_calls_retrieval():

    (
        service,
        retrieval_service,
        _,
        _,
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
        _,
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
        _,
        _,
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


def test_rag_service_returns_grounding_metadata():

    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert response.metadata["is_grounded"] is True

    assert (
        response.metadata["grounding_reason"]
        == "Test answer is grounded."
    )


def test_rag_service_returns_source_validation_metadata():

    (
        service,
        _,
        _,
        _,
        _,
        _,
        source_validator,
    ) = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert source_validator.called is True

    assert source_validator.received_context is not None

    assert (
        len(source_validator.received_sources)
        == 1
    )

    assert (
        response.metadata["source_validation"]
        is True
    )


def test_rag_request_rejects_empty_query():

    with pytest.raises(ValueError):

        RAGRequest(
            query="   "
        )


def test_rag_service_returns_multiple_sources():

    service, _, _, _, _, _, _ = create_service()

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

    service, _, context_builder, _, _, _, _ = create_service()

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

    service, _, _, _, _, _, _ = create_service()

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

    answer_validator = FakeAnswerValidator()

    source_validator = FakeSourceValidator()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        answer_validator=answer_validator,
        source_validator=source_validator,
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
    # 2. Retrieval returned no results
    # --------------------------------------------------------

    assert (
        context_builder.received_results
        == []
    )

    # --------------------------------------------------------
    # 3. Context builder was called
    # --------------------------------------------------------

    assert context_builder.called is True

    # --------------------------------------------------------
    # 4. Prompt builder should NOT be called
    # --------------------------------------------------------

    assert prompt_builder.called is False

    assert prompt_builder.received_context is None

    # --------------------------------------------------------
    # 5. LLM should NOT be called
    # --------------------------------------------------------

    assert llm_service.called is False

    # --------------------------------------------------------
    # 6. Controlled no-context answer
    # --------------------------------------------------------

    assert (
        response.answer
        == "I couldn't find relevant information "
        "in the provided documents."
    )

    # --------------------------------------------------------
    # 7. There should be no sources
    # --------------------------------------------------------

    assert response.sources == []

    # --------------------------------------------------------
    # 8. Metadata should report zero retrieved chunks
    # --------------------------------------------------------

    assert (
        response.metadata["retrieved_chunks"]
        == 0
    )

    # --------------------------------------------------------
    # 9. Answer should NOT be considered grounded
    # --------------------------------------------------------

    assert response.metadata["is_grounded"] is False

    assert (
        response.metadata["grounding_reason"]
        == "No retrieved context is available."
    )


def test_rag_service_passes_generated_prompt_to_llm():

    service, _, _, _, llm_service, _, _ = create_service()

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert llm_service.called is True

    assert llm_service.received_prompt is not None

    assert (
        "What is machine learning?"
        in llm_service.received_prompt
    )

    assert (
        response.answer
        == "Machine learning enables systems "
        "to learn patterns from data."
    )

def test_rag_service_validates_sources():

    retrieval_service = FakeRetrievalService()
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()
    llm_service = FakeLLMService()
    answer_validator = FakeAnswerValidator()
    source_validator = FakeSourceValidator()

    service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        answer_validator=answer_validator,
        source_validator=source_validator,
    )

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert source_validator.called is True

    assert (
        source_validator.received_context
        is not None
    )

    assert (
        len(source_validator.received_sources)
        == 1
    )

    assert (
        response.metadata["source_validation"]
        is True
    )

