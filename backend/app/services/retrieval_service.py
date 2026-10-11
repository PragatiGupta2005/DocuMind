
from app.embeddings.embedding_service import EmbeddingService
from app.schemas.search_result_schema import SearchResultSchema
from app.vector_store.qdrant_store import QdrantVectorStore
from app.core.settings import MIN_RELEVANCE_SCORE


class RetrievalService:
    """
    Handles query embedding and vector similarity retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        """Initialize the retrieval service."""

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    @staticmethod
    def _deduplicate_results(
        results: list[SearchResultSchema],
    ) -> list[SearchResultSchema]:
        """
        Keep the highest-scoring result for each point_id.

        Results are returned in descending score order.
        Distinct points are retained even if their text is identical.
        """

        best_results: dict[str, SearchResultSchema] = {}

        for result in results:
            existing = best_results.get(result.point_id)

            if (
                existing is None
                or result.score > existing.score
            ):
                best_results[result.point_id] = result

        return sorted(
            best_results.values(),
            key=lambda result: result.score,
            reverse=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResultSchema]:
        """
        Embed the query, retrieve candidates, deduplicate them,
        and filter results using the configured relevance threshold.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.embedding_service.embed_text(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
        )

        unique_results = self._deduplicate_results(results)

        return [
            result
            for result in unique_results
            if result.score >= MIN_RELEVANCE_SCORE
        ]
