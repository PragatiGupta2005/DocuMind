from pydantic import BaseModel, Field


class DocumentStatisticsSchema(BaseModel):
    """
    Aggregate statistics for uploaded documents.
    """

    total_documents: int = Field(
        ...,
        ge=0,
        description="Total number of registered documents"
    )

    total_chunks: int = Field(
        ...,
        ge=0,
        description="Total number of chunks across all documents"
    )

    total_size: int = Field(
        ...,
        ge=0,
        description="Total size of all documents in bytes"
    )

    file_types: dict[str, int] = Field(
        default_factory=dict,
        description="Number of documents grouped by file type"
    )