from abc import ABC, abstractmethod

from app.schemas.document_schema import DocumentSchema
from app.schemas.chunk_schema import ChunkSchema


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    @abstractmethod
    def chunk(
        self,
        document: DocumentSchema
    ) -> list[ChunkSchema]:
        """
        Split a document into chunks.
        """
        pass