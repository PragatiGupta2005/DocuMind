from app.schemas.rag_context_schema import RAGContext
from app.schemas.rag_schema import SourceReference


class SourceValidator:

    def validate(
        self,
        context: RAGContext,
        sources: list[SourceReference],
    ) -> bool:

        if not context.chunks:
            return sources == []

        if len(context.chunks) != len(sources):
            return False

        for chunk, source in zip(context.chunks, sources):

            if chunk.document_id != source.document_id:
                return False

            if chunk.document_name != source.document_name:
                return False

            if chunk.chunk_id != source.chunk_id:
                return False

            if chunk.score != source.score:
                return False

        return True