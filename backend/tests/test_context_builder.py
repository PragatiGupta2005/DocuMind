from app.rag.context_builder import ContextBuilder
from app.schemas.search_result_schema import SearchResultSchema


def create_result(
    point_id="point-001",
    score=0.95,
    document_id="doc-001",
    document_name="test.pdf",
    chunk_id=1,
    text="Test document content.",
):
    return SearchResultSchema(
        point_id=point_id,
        score=score,
        payload={
            "document_id": document_id,
            "document_name": document_name,
            "chunk_id": chunk_id,
            "text": text,
        },
    )


def test_context_builder_handles_empty_results():

    builder = ContextBuilder()

    context = builder.build([])

    assert context.chunks == []

    assert context.formatted_context == ""


def test_context_builder_creates_single_context_chunk():

    builder = ContextBuilder()

    result = create_result()

    context = builder.build([result])

    assert len(context.chunks) == 1

    chunk = context.chunks[0]

    assert chunk.document_id == "doc-001"
    assert chunk.document_name == "test.pdf"
    assert chunk.chunk_id == 1
    assert chunk.text == "Test document content."
    assert chunk.score == 0.95


def test_context_builder_preserves_multiple_results():

    builder = ContextBuilder()

    results = [
        create_result(
            point_id="point-001",
            score=0.95,
            chunk_id=1,
            text="First chunk.",
        ),
        create_result(
            point_id="point-002",
            score=0.85,
            chunk_id=2,
            text="Second chunk.",
        ),
    ]

    context = builder.build(results)

    assert len(context.chunks) == 2

    assert context.chunks[0].chunk_id == 1
    assert context.chunks[1].chunk_id == 2


def test_context_builder_preserves_result_order():

    builder = ContextBuilder()

    results = [
        create_result(
            point_id="point-001",
            score=0.95,
            text="Highest relevance.",
        ),
        create_result(
            point_id="point-002",
            score=0.80,
            text="Lower relevance.",
        ),
    ]

    context = builder.build(results)

    assert (
        context.chunks[0].text
        == "Highest relevance."
    )

    assert (
        context.chunks[1].text
        == "Lower relevance."
    )


def test_context_builder_formats_context():

    builder = ContextBuilder()

    result = create_result(
        score=0.912345,
        text="Machine learning is useful.",
    )

    context = builder.build([result])

    assert "[Source 1]" in context.formatted_context
    assert "Document: test.pdf" in context.formatted_context
    assert "Chunk ID: 1" in context.formatted_context
    assert "Relevance Score: 0.9123" in context.formatted_context
    assert "Machine learning is useful." in context.formatted_context