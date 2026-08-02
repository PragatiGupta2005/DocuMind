from fastapi import FastAPI

from app.api.upload import router as upload_router

app = FastAPI(
    title="DocuMind API",
    description="Enterprise Multi Document RAG System",
    version="1.0.0"
)

app.include_router(upload_router)


@app.get("/")
async def home():
    return {
        "message": "Welcome to DocuMind API 🚀"
    }