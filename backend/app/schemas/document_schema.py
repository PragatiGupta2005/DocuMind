from typing import Any

from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    """
    Represents a processed document.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document"
    )

    filename: str = Field(
        ...,
        description="Document filename"
    )

    file_type: str = Field(
        ...,
        description="Document file type"
    )

    text: str = Field(
        ...,
        description="Extracted document text"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )