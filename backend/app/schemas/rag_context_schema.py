from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """
    Represents a single retrieved chunk prepared for RAG context.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document",
    )

    document_name: str = Field(
        ...,
        description="Original document filename",
    )

    chunk_id: int = Field(
        ...,
        description="Sequential chunk number within the document",
    )

    text: str = Field(
        ...,
        description="Retrieved chunk text",
    )

    score: float = Field(
        ...,
        description="Similarity score returned by vector search",
    )


class RAGContext(BaseModel):
    """
    Represents the complete context supplied to the RAG pipeline.
    """

    chunks: list[ContextChunk] = Field(
        default_factory=list,
        description="Retrieved chunks used as context",
    )

    formatted_context: str = Field(
        default="",
        description="Formatted text representation of the context",
    )