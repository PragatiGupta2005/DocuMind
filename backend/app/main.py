from fastapi import FastAPI
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router

app = FastAPI(
    title="DocuMind API",
    description="Document-based Retrieval Augmented Generation system",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to DocuMind API",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

app.include_router(documents_router)
app.include_router(rag_router)