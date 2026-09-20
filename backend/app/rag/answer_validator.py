from app.schemas.answer_validation_schema import AnswerValidationResult
from app.schemas.rag_context_schema import RAGContext


class AnswerValidator:

    def validate(
        self,
        answer: str,
        context: RAGContext,
    ) -> AnswerValidationResult:

        if not answer or not answer.strip():
            return AnswerValidationResult(
                is_grounded=False,
                reason="Generated answer is empty.",
            )

        if not context.chunks:
            return AnswerValidationResult(
                is_grounded=False,
                reason="No retrieved context is available.",
            )

        return AnswerValidationResult(
            is_grounded=True,
            reason="Answer was generated with retrieved document context.",
        )