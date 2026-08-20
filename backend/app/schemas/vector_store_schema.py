from typing import Any

from pydantic import BaseModel, Field, model_validator


class VectorStoreSchema(BaseModel):
    """
    Represents a vector record that can be stored
    in the vector database.
    """

    point_id: str = Field(
        ...,
        description="Unique identifier of the vector-store point"
    )

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
        description="Name of the embedding model"
    )

    dimensions: int = Field(
        ...,
        description="Number of dimensions in the vector"
    )

    vector: list[float] = Field(
        ...,
        description="Embedding vector"
    )

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata stored with the vector"
    )

    @model_validator(mode="after")
    def validate_dimensions(self):
        """
        Ensure the declared dimensions match
        the actual vector length.
        """

        if len(self.vector) != self.dimensions:
            raise ValueError(
                "Vector dimensions do not match "
                "the declared dimensions."
            )

        return self