import logging
import time

from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.rag.answer_validator import AnswerValidator
from app.rag.source_validator import SourceValidator
from app.schemas.rag_schema import (
    RAGRequest,
    RAGResponse,
    SourceReference,
)
from app.services.retrieval_service import RetrievalService
from app.llm.llm_service import LLMService
from app.exceptions.rag_exceptions import (
    RetrievalError,
    LLMError,
)


logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestrates the complete Retrieval-Augmented
    Generation pipeline.
    """

    def __init__(
        self,
        retrieval_service,
        context_builder,
        prompt_builder,
        llm_service,
        answer_validator,
        source_validator,
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.answer_validator = answer_validator
        self.source_validator = source_validator

    def generate(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        """
        Execute the complete RAG pipeline.
        """

        # --------------------------------------------------
        # 1. Validate query
        # --------------------------------------------------

        if not request.query or not request.query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------
        # 8.4.1 RAG Request Logging
        # --------------------------------------------------

        logger.info(
            "RAG request received | query_length=%d | "
            "top_k=%d | document_id=%s",
            len(request.query),
            request.top_k,
            request.document_id,
        )

        # --------------------------------------------------
        # 8.5.4 Pipeline Performance
        # Start total RAG pipeline timer
        # --------------------------------------------------

        pipeline_start_time = time.perf_counter()

        logger.info(
            "RAG pipeline started | top_k=%d | document_id=%s",
            request.top_k,
            request.document_id,
        )

        # --------------------------------------------------
        # 2. Retrieve relevant chunks
        # --------------------------------------------------

        retrieval_start_time = time.perf_counter()

        try:
            results = self.retrieval_service.retrieve(
                query=request.query,
                top_k=request.top_k,
                document_id=request.document_id,
            )

            retrieval_latency = (
                time.perf_counter() - retrieval_start_time
            )

            logger.info(
                "Retrieval completed | "
                "retrieved_chunks=%d | "
                "top_k=%d | "
                "document_id=%s | "
                "latency=%.4fs",
                len(results),
                request.top_k,
                request.document_id,
                retrieval_latency,
            )

        except Exception as exc:
            retrieval_latency = (
                time.perf_counter() - retrieval_start_time
            )

            pipeline_latency = (
                time.perf_counter() - pipeline_start_time
            )

            logger.error(
                "Retrieval failed | "
                "latency=%.4fs | "
                "pipeline_latency=%.4fs | "
                "error=%s",
                retrieval_latency,
                pipeline_latency,
                exc,
            )

            raise RetrievalError(str(exc)) from exc

        # --------------------------------------------------
        # 3. Build context
        # --------------------------------------------------

        context = self.context_builder.build(results)

        # --------------------------------------------------
        # 4. Handle no-context case
        # --------------------------------------------------

        if not context.chunks:

            pipeline_latency = (
                time.perf_counter() - pipeline_start_time
            )

            logger.info(
                "No relevant context found | "
                "top_k=%d | "
                "document_id=%s | "
                "pipeline_latency=%.4fs",
                request.top_k,
                request.document_id,
                pipeline_latency,
            )

            return RAGResponse(
                answer=(
                    "I couldn't find relevant information "
                    "in the provided documents."
                ),
                sources=[],
                metadata={
                    "model_name": (
                        self.llm_service.get_model_name()
                    ),
                    "retrieved_chunks": 0,
                    "is_grounded": False,
                    "grounding_reason": (
                        "No retrieved context is available."
                    ),
                    "source_validation": True,
                    "pipeline_latency": pipeline_latency,
                },
            )

        # --------------------------------------------------
        # 5. Build LLM prompt
        # --------------------------------------------------

        prompt = self.prompt_builder.build(
            query=request.query,
            context=context,
        )

        # --------------------------------------------------
        # 6. Generate answer
        # --------------------------------------------------

        logger.info(
            "LLM generation started | model=%s",
            self.llm_service.get_model_name(),
        )

        llm_start_time = time.perf_counter()

        try:
            answer = self.llm_service.generate(prompt)

        except Exception as exc:
            llm_latency = (
                time.perf_counter() - llm_start_time
            )

            pipeline_latency = (
                time.perf_counter() - pipeline_start_time
            )

            logger.error(
                "LLM generation failed | "
                "model=%s | "
                "latency=%.4fs | "
                "pipeline_latency=%.4fs | "
                "error=%s",
                self.llm_service.get_model_name(),
                llm_latency,
                pipeline_latency,
                exc,
            )

            raise LLMError(str(exc)) from exc

        llm_latency = (
            time.perf_counter() - llm_start_time
        )

        logger.info(
            "LLM generation completed | "
            "model=%s | "
            "latency=%.4fs",
            self.llm_service.get_model_name(),
            llm_latency,
        )

        # --------------------------------------------------
        # 7. Validate answer grounding
        # --------------------------------------------------

        validation = self.answer_validator.validate(
            answer=answer,
            context=context,
        )

        # --------------------------------------------------
        # 8. Build source references
        # --------------------------------------------------

        sources = self._build_sources(context)

        # --------------------------------------------------
        # 9. Validate source references
        # --------------------------------------------------

        source_validation = self.source_validator.validate(
            context=context,
            sources=sources,
        )

        # --------------------------------------------------
        # 10. Calculate total pipeline latency
        # --------------------------------------------------

        pipeline_latency = (
            time.perf_counter() - pipeline_start_time
        )

        logger.info(
            "RAG pipeline completed | "
            "latency=%.4fs | "
            "retrieved_chunks=%d",
            pipeline_latency,
            len(context.chunks),
        )

        # --------------------------------------------------
        # 11. Return structured response
        # --------------------------------------------------

        return RAGResponse(
            answer=answer,
            sources=sources,
            metadata={
                "model_name": (
                    self.llm_service.get_model_name()
                ),
                "retrieved_chunks": len(
                    context.chunks
                ),
                "is_grounded": validation.is_grounded,
                "grounding_reason": validation.reason,
                "source_validation": source_validation,
                "pipeline_latency": pipeline_latency,
            },
        )

    def _build_sources(
        self,
        context,
    ) -> list[SourceReference]:
        """
        Convert context chunks into source references.
        """

        sources = []

        for chunk in context.chunks:
            sources.append(
                SourceReference(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                )
            )

        return sources