from app.embeddings.embedding_service import EmbeddingService
from app.schemas.search_result_schema import SearchResultSchema
from app.vector_store.qdrant_store import QdrantVectorStore


class RetrievalService:
    """
    Handles query embedding and vector similarity retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        """
        Initialize the retrieval service.
        """

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResultSchema]:
        """
        Generate an embedding for the query and retrieve
        the most relevant chunks from the vector store.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_vector = (
            self.embedding_service.embed_text(query)
        )

        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
        )