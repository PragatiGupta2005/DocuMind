from typing import Any

from pydantic import BaseModel, Field


class EmbeddingSchema(BaseModel):
    """
    Represents an embedding generated from a document chunk.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document"
    )

    chunk_uuid: str = Field(
        ...,
        description="Unique identifier of the source chunk"
    )

    model_name: str = Field(
        ...,
        description="Name of the embedding model used"
    )

    dimensions: int = Field(
        ...,
        description="Number of dimensions in the embedding vector"
    )

    vector: list[float] = Field(
        ...,
        description="Numerical embedding vector"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the embedding"
    )