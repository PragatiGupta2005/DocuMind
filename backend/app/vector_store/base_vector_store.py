from abc import ABC, abstractmethod

from app.schemas.vector_store_schema import VectorStoreSchema


class BaseVectorStore(ABC):
    """
    Abstract interface for vector storage and retrieval.
    """

    @abstractmethod
    def add(
        self,
        records: list[VectorStoreSchema]
    ) -> None:
        """
        Add vector records to the vector store.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[VectorStoreSchema]:
        """
        Search for the most similar vectors.
        """
        pass

    @abstractmethod
    def delete(
        self,
        document_id: str
    ) -> None:
        """
        Delete all vectors belonging to a document.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of stored vectors.
        """
        pass