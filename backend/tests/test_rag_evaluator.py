from app.evaluation.rag_evaluator import (
    RAGEvaluator,
)
from app.schemas.rag_schema import (
    RAGResponse,
    SourceReference,
)


def create_response():

    return RAGResponse(
        answer=(
            "Machine learning enables "
            "systems to learn from data."
        ),
        sources=[
            SourceReference(
                document_id="doc-001",
                document_name="machine-learning.pdf",
                chunk_id=1,
                score=0.95,
            ),
            SourceReference(
                document_id="doc-002",
                document_name="database.pdf",
                chunk_id=1,
                score=0.72,
            ),
        ],
        metadata={
            "retrieved_chunks": 2,
        },
    )


def test_evaluator_finds_expected_document():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate_retrieval(
        response=response,
        expected_document_id="doc-001",
        top_k=5,
    )

    assert result.expected_document_found is True


def test_evaluator_finds_expected_chunk():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate_retrieval(
        response=response,
        expected_document_id="doc-001",
        expected_chunk_id=1,
        top_k=5,
    )

    assert result.expected_chunk_found is True


def test_evaluator_detects_missing_document():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate_retrieval(
        response=response,
        expected_document_id="doc-999",
        top_k=5,
    )

    assert result.expected_document_found is False


def test_evaluator_respects_top_k():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate_retrieval(
        response=response,
        top_k=2,
    )

    assert result.top_k_respected is True


def test_evaluator_detects_top_k_violation():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate_retrieval(
        response=response,
        top_k=1,
    )

    assert result.top_k_respected is False


def test_evaluator_returns_perfect_score():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate(
        response=response,
        expected_document_id="doc-001",
        expected_chunk_id=1,
        top_k=5,
    )

    assert result.retrieval_score == 1.0

    assert result.passed is True


def test_evaluator_returns_failure_score():

    evaluator = RAGEvaluator()

    response = create_response()

    result = evaluator.evaluate(
        response=response,
        expected_document_id="doc-999",
        expected_chunk_id=99,
        top_k=5,
    )

    assert result.retrieval_score < 1.0

    assert result.passed is False