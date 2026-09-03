from fastapi import APIRouter
from app.schemas.rag_query_request_schema import RAGQueryRequest
from app.schemas.rag_query_response_schema import RAGQueryResponse

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post("/query", response_model=RAGQueryResponse)
async def query(request: RAGQueryRequest):
    pass