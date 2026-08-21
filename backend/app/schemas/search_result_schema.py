from typing import Any

from pydantic import BaseModel, Field


class SearchResultSchema(BaseModel):
    """
    Represents a single vector similarity search result.
    """

    point_id: str = Field(
        ...,
        description="Unique identifier of the matched vector"
    )

    score: float = Field(
        ...,
        description="Similarity score returned by the vector database"
    )

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Payload associated with the matched vector"
    )