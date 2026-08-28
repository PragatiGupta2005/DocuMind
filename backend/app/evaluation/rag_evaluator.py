from app.schemas.evaluation_schema import (
    RetrievalEvaluation,
    RAGEvaluationResult,
)
from app.schemas.rag_schema import RAGResponse


class RAGEvaluator:
    """
    Evaluates deterministic properties of a RAG response.
    """

    def evaluate_retrieval(
        self,
        response: RAGResponse,
        expected_document_id: str | None = None,
        expected_chunk_id: int | None = None,
        top_k: int = 5,
    ) -> RetrievalEvaluation:

        sources = response.sources

        retrieved_count = len(sources)

        expected_document_found = True

        if expected_document_id is not None:

            expected_document_found = any(
                source.document_id
                == expected_document_id
                for source in sources
            )

        expected_chunk_found = True

        if (
            expected_document_id is not None
            and expected_chunk_id is not None
        ):

            expected_chunk_found = any(
                source.document_id
                == expected_document_id
                and source.chunk_id
                == expected_chunk_id
                for source in sources
            )

        top_k_respected = (
            retrieved_count <= top_k
        )

        return RetrievalEvaluation(
            retrieved_count=retrieved_count,
            expected_document_found=(
                expected_document_found
            ),
            expected_chunk_found=(
                expected_chunk_found
            ),
            top_k_respected=(
                top_k_respected
            ),
        )

    def evaluate(
        self,
        response: RAGResponse,
        expected_document_id: str | None = None,
        expected_chunk_id: int | None = None,
        top_k: int = 5,
    ) -> RAGEvaluationResult:

        retrieval = self.evaluate_retrieval(
            response=response,
            expected_document_id=expected_document_id,
            expected_chunk_id=expected_chunk_id,
            top_k=top_k,
        )

        checks = [
            retrieval.expected_document_found,
            retrieval.expected_chunk_found,
            retrieval.top_k_respected,
        ]

        retrieval_score = (
            sum(checks) / len(checks)
        )

        passed = (
            retrieval_score == 1.0
        )

        return RAGEvaluationResult(
            retrieval=retrieval,
            retrieval_score=retrieval_score,
            passed=passed,
            details={
                "evaluation_type": "deterministic",
                "checks_passed": sum(checks),
                "total_checks": len(checks),
            },
        )