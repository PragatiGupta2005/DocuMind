from app.embeddings.embedding_service import EmbeddingService
from app.schemas.chunk_schema import ChunkSchema
import pytest
def create_test_chunk(
    chunk_id: int,
    text: str
) -> ChunkSchema:

    return ChunkSchema(
        document_id="doc_001",
        chunk_uuid=f"chunk-{chunk_id}",
        chunk_id=chunk_id,
        document_name="AI.pdf",
        text=text,
        start_index=0,
        end_index=len(text),
        metadata={
            "source": "test"
        }
    )

def test_embed_single_chunk():

    service = EmbeddingService()
    chunk = create_test_chunk(chunk_id=1,text="Artificial Intelligence is amazing.")
    result = service.embed_chunk(chunk)

    assert result.document_id == "doc_001"
    assert result.chunk_uuid == "chunk-1"

    assert result.model_name != ""
    assert result.dimensions > 0

    assert isinstance(result.vector, list)
    assert len(result.vector) == result.dimensions

    assert result.metadata["chunk_id"] == 1
    assert result.metadata["document_name"] == "AI.pdf"

def test_embed_multiple_chunks():
    service = EmbeddingService()
    chunks = [
        create_test_chunk(
            chunk_id=1,
            text="Artificial Intelligence is amazing."
        ),
        create_test_chunk(
            chunk_id=2,
            text="Machine Learning is a subset of AI."
        ),
        create_test_chunk(
            chunk_id=3,
            text="Deep Learning uses Neural Networks."
        )
    ]

    results = service.embed_chunks(chunks)
    assert len(results) == 3
    for chunk, embedding in zip(
        chunks,
        results
    ):
        assert (
            embedding.document_id
            == chunk.document_id
        )

        assert (
            embedding.chunk_uuid
            == chunk.chunk_uuid
        )

        assert embedding.dimensions > 0

        assert (
            len(embedding.vector)
            == embedding.dimensions
        )

def test_empty_chunk_list():
    service = EmbeddingService()
    result = service.embed_chunks([])
    assert result == []

def test_empty_chunk_text():

    service = EmbeddingService()
    chunk = create_test_chunk(
        chunk_id=1,
        text=""
    )

    with pytest.raises(
        ValueError,
        match="Chunk text cannot be empty."
    ):
        service.embed_chunk(chunk)

def test_whitespace_chunk_text():

    service = EmbeddingService()

    chunk = create_test_chunk(
        chunk_id=1,
        text="   "
    )

    with pytest.raises(
        ValueError,
        match="Chunk text cannot be empty."
    ):
        service.embed_chunk(chunk)

def test_empty_text_in_batch():

    service = EmbeddingService()

    chunks = [
        create_test_chunk(
            chunk_id=1,
            text="Artificial Intelligence is amazing."
        ),
        create_test_chunk(
            chunk_id=2,
            text=""
        )
    ]

    with pytest.raises(
        ValueError,
        match="Chunk 2 contains empty text."
    ):
        service.embed_chunks(chunks)

class FakeProvider:

    def embed_batch(self, texts):

        # Intentionally return fewer vectors
        return [
            [0.1, 0.2, 0.3]
        ]

    def embed(self, text):

        return [0.1, 0.2, 0.3]

    def get_model_name(self):

        return "fake-model"

def test_embedding_count_mismatch():

    service = EmbeddingService()

    service.provider = FakeProvider()

    chunks = [
        create_test_chunk(
            chunk_id=1,
            text="First chunk"
        ),
        create_test_chunk(
            chunk_id=2,
            text="Second chunk"
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="Number of generated embeddings does not match"
    ):
        service.embed_chunks(chunks)

def test_local_embedding_dimensions():
    from app.embeddings.local_embedding import LocalEmbedding

    embedding = LocalEmbedding()

    assert embedding.get_dimensions() == 384

def test_api_embedding_dimensions():
    from app.embeddings.api_embedding import APIEmbedding

    embedding = APIEmbedding.__new__(
        APIEmbedding
    )

    embedding.model_name = "gemini-embedding-001"

    assert embedding.get_dimensions() == 3072