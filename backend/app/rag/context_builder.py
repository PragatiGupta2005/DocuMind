from app.schemas.rag_context_schema import (
    ContextChunk,
    RAGContext,
)
from app.schemas.search_result_schema import SearchResultSchema


class ContextBuilder:
    """
    Converts retrieved search results into
    structured and LLM-ready RAG context.
    """

    def build(
        self,
        results: list[SearchResultSchema],
    ) -> RAGContext:
        """
        Build RAG context from retrieved search results.
        """

        if not results:
            return RAGContext(
                chunks=[],
                formatted_context="",
            )

        context_chunks = []

        for result in results:

            payload = result.payload

            context_chunk = ContextChunk(
                document_id=payload["document_id"],
                document_name=payload["document_name"],
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                score=result.score,
            )

            context_chunks.append(
                context_chunk
            )

        formatted_context = self._format_context(
            context_chunks
        )

        return RAGContext(
            chunks=context_chunks,
            formatted_context=formatted_context,
        )

    def _format_context(
        self,
        chunks: list[ContextChunk],
    ) -> str:
        """
        Convert context chunks into structured text.
        """

        sections = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            section = (
                f"[Source {index}]\n"
                f"Document: {chunk.document_name}\n"
                f"Document ID: {chunk.document_id}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Relevance Score: {chunk.score:.4f}\n"
                f"Content:\n"
                f"{chunk.text}"
            )

            sections.append(section)

        return "\n\n".join(sections)