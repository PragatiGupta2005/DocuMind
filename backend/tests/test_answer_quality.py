from app.rag.rag_service import RAGService
from app.schemas.rag_schema import RAGRequest
from app.schemas.answer_validation_schema import AnswerValidationResult


# ============================================================
# Fake Retrieval Service
# ============================================================

class FakeRetrievalService:

    def __init__(self, results=None):
        self.results = results
        self.called = False

    def retrieve(
        self,
        query,
        top_k,
        document_id=None,
    ):
        self.called = True
        return self.results


# ============================================================
# Fake Context Builder
# ============================================================

class FakeContextBuilder:

    def __init__(self, context):
        self.context = context

    def build(self, results):
        return self.context


# ============================================================
# Fake Prompt Builder
# ============================================================

class FakePromptBuilder:

    def build(self, query, context):
        return (
            f"Question: {query}\n"
            f"Context: {context.formatted_context}"
        )


# ============================================================
# Fake LLM Service
# ============================================================

class FakeLLMService:

    def __init__(self, answer):
        self.answer = answer
        self.called = False

    def generate(self, prompt):
        self.called = True
        return self.answer

    def get_model_name(self):
        return "fake-llm"


# ============================================================
# Fake Answer Validator
# ============================================================

class FakeAnswerValidator:

    def __init__(self, is_grounded=True):
        self.is_grounded = is_grounded

    def validate(self, answer, context):
        return AnswerValidationResult(
            is_grounded=self.is_grounded,
            reason=(
                "Answer is grounded in retrieved context."
                if self.is_grounded
                else "Answer is not grounded."
            ),
        )


# ============================================================
# Fake Source Validator
# ============================================================

class FakeSourceValidator:

    def __init__(self, is_valid=True):
        self.is_valid = is_valid

    def validate(self, context, sources):
        return self.is_valid


# ============================================================
# Test Helpers
# ============================================================

def create_context():
    from app.schemas.rag_context_schema import (
        ContextChunk,
        RAGContext,
    )

    chunk = ContextChunk(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=1,
        text=(
            "Machine learning enables systems "
            "to learn patterns from data."
        ),
        score=0.95,
    )

    return RAGContext(
        chunks=[chunk],
        formatted_context=(
            "Machine learning enables systems "
            "to learn patterns from data."
        ),
    )


def create_empty_context():
    from app.schemas.rag_context_schema import RAGContext

    return RAGContext(
        chunks=[],
        formatted_context="",
    )


def create_retrieval_result():
    from app.schemas.search_result_schema import (
        SearchResultSchema,
    )

    return [
        SearchResultSchema(
            point_id="point-001",
            score=0.95,
            payload={
                "document_id": "doc-001",
                "document_name": "test.pdf",
                "chunk_id": 1,
                "text": (
                    "Machine learning enables systems "
                    "to learn patterns from data."
                ),
            },
        )
    ]


def create_service(
    context,
    llm_answer=(
        "Machine learning enables systems "
        "to learn patterns from data."
    ),
    is_grounded=True,
    source_validation=True,
):

    retrieval_service = FakeRetrievalService(
        results=create_retrieval_result()
        if context.chunks
        else []
    )

    context_builder = FakeContextBuilder(
        context=context
    )

    prompt_builder = FakePromptBuilder()

    llm_service = FakeLLMService(
        answer=llm_answer
    )

    answer_validator = FakeAnswerValidator(
        is_grounded=is_grounded
    )

    source_validator = FakeSourceValidator(
        is_valid=source_validation
    )

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
        llm_service,
    )


# ============================================================
# 1. Grounded Answer
# ============================================================

def test_answer_quality_grounded_answer():

    service, _, _ = create_service(
        context=create_context(),
        is_grounded=True,
    )

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert response.metadata["is_grounded"] is True

    assert (
        response.metadata["grounding_reason"]
        == "Answer is grounded in retrieved context."
    )


# ============================================================
# 2. Ungrounded Answer
# ============================================================

def test_answer_quality_ungrounded_answer():

    service, _, _ = create_service(
        context=create_context(),
        is_grounded=False,
    )

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert response.metadata["is_grounded"] is False

    assert (
        response.metadata["grounding_reason"]
        == "Answer is not grounded."
    )


# ============================================================
# 3. No Context
# ============================================================

def test_answer_quality_no_context():

    service, _, llm_service = create_service(
        context=create_empty_context(),
    )

    request = RAGRequest(
        query="What is quantum computing?"
    )

    response = service.generate(request)

    assert (
        response.answer
        == "I couldn't find relevant information "
           "in the provided documents."
    )

    assert response.sources == []

    assert (
        response.metadata["retrieved_chunks"]
        == 0
    )

    assert response.metadata["is_grounded"] is False

    assert llm_service.called is False


# ============================================================
# 4. Source Validation
# ============================================================

def test_answer_quality_source_validation():

    service, _, _ = create_service(
        context=create_context(),
        source_validation=True,
    )

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert (
        response.metadata["source_validation"]
        is True
    )


# ============================================================
# 5. Invalid Source Validation
# ============================================================

def test_answer_quality_invalid_source_validation():

    service, _, _ = create_service(
        context=create_context(),
        source_validation=False,
    )

    request = RAGRequest(
        query="What is machine learning?"
    )

    response = service.generate(request)

    assert (
        response.metadata["source_validation"]
        is False
    )


# ============================================================
# 6. Sources Exist When Context Exists
# ============================================================

def test_answer_quality_sources_exist_with_context():

    service, _, _ = create_service(
        context=create_context(),
    )

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