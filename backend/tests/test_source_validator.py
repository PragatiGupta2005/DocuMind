from app.rag.source_validator import SourceValidator
from app.schemas.rag_context_schema import ContextChunk, RAGContext
from app.schemas.rag_schema import SourceReference


def create_context():

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
        formatted_context="Machine learning learns patterns from data.",
    )


def create_source():

    return SourceReference(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=1,
        score=0.95,
    )


def test_source_matches_context():

    validator = SourceValidator()

    context = create_context()
    sources = [create_source()]

    result = validator.validate(
        context=context,
        sources=sources,
    )

    assert result is True


def test_source_document_id_mismatch():

    validator = SourceValidator()

    context = create_context()

    source = SourceReference(
        document_id="wrong-document",
        document_name="test.pdf",
        chunk_id=1,
        score=0.95,
    )

    result = validator.validate(
        context=context,
        sources=[source],
    )

    assert result is False


def test_source_chunk_id_mismatch():

    validator = SourceValidator()

    context = create_context()

    source = SourceReference(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=99,
        score=0.95,
    )

    result = validator.validate(
        context=context,
        sources=[source],
    )

    assert result is False


def test_source_score_mismatch():

    validator = SourceValidator()

    context = create_context()

    source = SourceReference(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=1,
        score=0.50,
    )

    result = validator.validate(
        context=context,
        sources=[source],
    )

    assert result is False


def test_no_context_requires_no_sources():

    validator = SourceValidator()

    context = RAGContext(
        chunks=[],
        formatted_context="",
    )

    result = validator.validate(
        context=context,
        sources=[],
    )

    assert result is True


def test_no_context_with_sources_is_invalid():

    validator = SourceValidator()

    context = RAGContext(
        chunks=[],
        formatted_context="",
    )

    source = create_source()

    result = validator.validate(
        context=context,
        sources=[source],
    )

    assert result is False