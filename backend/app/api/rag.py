from fastapi import APIRouter

from app.embeddings.embedding_service import EmbeddingService
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_service import RAGService
from app.schemas.rag_schema import RAGRequest, RAGResponse
from app.services.retrieval_service import RetrievalService
from app.vector_store.collection_config import LOCAL_COLLECTION_NAME
from app.vector_store.qdrant_store import QdrantVectorStore
from app.llm.llm_service import LLMService


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


def create_rag_service() -> RAGService:
    """
    Create and configure the complete RAG service.
    """

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore(
        collection_name=LOCAL_COLLECTION_NAME
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    prompt_builder = PromptBuilder()

    llm_service = LLMService()

    return RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )


@router.post(
    "/query",
    response_model=RAGResponse,
)
async def query(request: RAGRequest):
    """
    Execute the RAG pipeline for a user query.
    """

    rag_service = create_rag_service()

    return rag_service.generate(request)