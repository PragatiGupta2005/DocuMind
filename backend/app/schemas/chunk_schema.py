from typing import Any

from pydantic import BaseModel, Field


class ChunkSchema(BaseModel):
    """
    Represents a single chunk extracted from a document.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document"
    )

    chunk_uuid: str = Field(
        ...,
        description="Globally unique identifier of the chunk"
    )

    chunk_id: int = Field(
        ...,
        description="Sequential chunk number within the document"
    )

    document_name: str = Field(
        ...,
        description="Original document filename"
    )

    text: str = Field(
        ...,
        description="Chunk text"
    )

    start_index: int = Field(
        ...,
        description="Starting character index of the chunk"
    )

    end_index: int = Field(
        ...,
        description="Ending character index of the chunk"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the chunk"
    )