from app.schemas.chunk_schema import ChunkSchema
from app.schemas.embedding_schema import EmbeddingSchema
from app.embeddings.embedding_factory import EmbeddingFactory

class EmbeddingService:
    """
    Handles embedding generation for document chunks.
    """
    def __init__(self):
        """
        Initialize the embedding provider using
        the configured embedding provider.
        """
        self.provider = EmbeddingFactory.get_provider()

    def embed_text(
        self,
        text: str
    ) -> list[float]:
        """
        Generate an embedding for raw text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        return self.provider.embed(text)

    def embed_chunk(
        self,
        chunk: ChunkSchema
    ) -> EmbeddingSchema:
        """
        Generate an embedding for a single chunk.
        """

        if not chunk.text or not chunk.text.strip():  #Single chunk
            raise ValueError("Chunk text cannot be empty.")

        vector = self.provider.embed(chunk.text)

        metadata = dict(chunk.metadata)
        metadata.update({
            "chunk_id": chunk.chunk_id,
            "document_name": chunk.document_name,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index
        })

        return EmbeddingSchema(
            document_id=chunk.document_id,
            chunk_uuid=chunk.chunk_uuid,
            model_name=self.provider.get_model_name(),
            dimensions=self.provider.get_dimensions(),
            vector=vector,
            metadata=metadata
        )

    def embed_chunks(
        self,
        chunks: list[ChunkSchema]
    ) -> list[EmbeddingSchema]:
        """
        Generate embeddings for multiple chunks.
        """

        if not chunks:
            return []

        texts = []

        for chunk in chunks:

            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id} contains empty text.")
            texts.append(chunk.text)

        vectors = self.provider.embed_batch(texts)
        if len(chunks) != len(vectors):
            raise RuntimeError(
                "Number of generated embeddings does not "
                "match number of input chunks."
            )
        embeddings = []
        for chunk, vector in zip(chunks,vectors):

            metadata = dict(chunk.metadata)

            metadata.update({
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index
            })

            embedding = EmbeddingSchema(
                document_id=chunk.document_id,
                chunk_uuid=chunk.chunk_uuid,
                model_name=self.provider.get_model_name(),
                dimensions=self.provider.get_dimensions(),
                vector=vector,
                metadata=metadata
            )

            embeddings.append(embedding)

        return embeddings