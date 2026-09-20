from app.rag.answer_validator import AnswerValidator
from app.schemas.rag_context_schema import ContextChunk, RAGContext


def create_context():
    chunk = ContextChunk(
        document_id="doc-1",
        document_name="machine_learning.txt",
        chunk_id=1,
        text="Machine learning enables systems to learn patterns from data.",
        score=0.85,
    )

    return RAGContext(
        chunks=[chunk],
        formatted_context=chunk.text,
    )


def test_answer_with_context_is_grounded():
    validator = AnswerValidator()

    result = validator.validate(
        answer="Machine learning enables systems to learn patterns from data.",
        context=create_context(),
    )

    assert result.is_grounded is True
    assert result.reason


def test_empty_answer_is_not_grounded():
    validator = AnswerValidator()

    result = validator.validate(
        answer="",
        context=create_context(),
    )

    assert result.is_grounded is False
    assert result.reason == "Generated answer is empty."


def test_no_context_is_not_grounded():
    validator = AnswerValidator()

    context = RAGContext(
        chunks=[],
        formatted_context="",
    )

    result = validator.validate(
        answer="Some generated answer.",
        context=context,
    )

    assert result.is_grounded is False
    assert result.reason == "No retrieved context is available."