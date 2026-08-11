from app.chunking.chunk_factory import ChunkFactory
from app.schemas.chunk_schema import ChunkSchema
from app.schemas.document_schema import DocumentSchema


class ChunkingService:
    """
    Handles the business logic for document chunking.
    """

    def chunk_document(
        self,
        document: DocumentSchema,
        strategy: str = "recursive"
    ) -> list[ChunkSchema]:
        """
        Chunk a document using the requested strategy.
        """

        chunker = ChunkFactory.get_chunker(strategy)

        chunks = chunker.chunk(document)

        return chunks