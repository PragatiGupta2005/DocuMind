from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.schemas.rag_schema import (
    RAGRequest,
    RAGResponse,
    SourceReference,
)
from app.services.retrieval_service import RetrievalService
from app.llm.llm_service import LLMService


class RAGService:
    """
    Orchestrates the complete Retrieval-Augmented
    Generation pipeline.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ):
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

    def generate(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        """
        Execute the complete RAG pipeline.
        """

        if not request.query or not request.query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------
        # 1. Retrieve relevant chunks
        # --------------------------------------------------

        results = self.retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        # --------------------------------------------------
        # 2. Build context
        # --------------------------------------------------

        context = self.context_builder.build(
            results
        )

        # --------------------------------------------------
        # 3. Build LLM prompt
        # --------------------------------------------------

        prompt = self.prompt_builder.build(
            query=request.query,
            context=context,
        )

        # --------------------------------------------------
        # 4. Generate answer
        # --------------------------------------------------

        answer = self.llm_service.generate(
            prompt
        )

        # --------------------------------------------------
        # 5. Build source references
        # --------------------------------------------------

        sources = self._build_sources(
            context
        )

        # --------------------------------------------------
        # 6. Return structured response
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