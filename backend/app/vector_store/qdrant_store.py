from qdrant_client.models import PointStruct
from app.schemas.vector_store_schema import VectorStoreSchema
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.qdrant_client import QdrantClientManager
from app.schemas.search_result_schema import SearchResultSchema

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation of the vector store.
    """

    def __init__(
        self,
        collection_name: str,
    ):
        self.collection_name = collection_name

        self.client = (
            QdrantClientManager()
            .get_client()
        )

    def add(
        self,
        records: list[VectorStoreSchema],
    ) -> None:
        """
        Store vector records in Qdrant.
        """

        points = []

        for record in records:

            point = PointStruct(
                id=record.point_id,
                vector=record.vector,
                payload={
                    "document_id": record.document_id,
                    "chunk_uuid": record.chunk_uuid,
                    "model_name": record.model_name,
                    "dimensions": record.dimensions,
                    **record.payload,
                },
            )

            points.append(point)

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
    self,
    query_vector: list[float],
    top_k: int = 5,
    ) -> list[SearchResultSchema]:
        """
        Search for the most similar vectors in Qdrant.
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            SearchResultSchema(
                point_id=str(result.id),
                score=result.score,
                payload=result.payload or {},
            )
            for result in results
        ]

    def delete(
        self,
        document_id: str,
    ) -> None:
        """
        Delete will be implemented in a later phase.
        """

        raise NotImplementedError(
            "Delete will be implemented later."
        )

    def count(self) -> int:
        """
        Return the number of stored vectors.
        """

        collection_info = (
            self.client.get_collection(
                self.collection_name
            )
        )

        return collection_info.points_count