from pydantic import BaseModel, Field


class RetrievalEvaluation(BaseModel):
    """
    Evaluation metrics for the retrieval stage.
    """

    retrieved_count: int = Field(
        ge=0
    )

    expected_document_found: bool

    expected_chunk_found: bool

    top_k_respected: bool


class RAGEvaluationResult(BaseModel):
    """
    Overall evaluation result for a RAG response.
    """

    retrieval: RetrievalEvaluation

    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    passed: bool

    details: dict = Field(
        default_factory=dict
    )