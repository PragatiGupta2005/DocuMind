from fastapi import FastAPI


app = FastAPI(
    title="DocuMind API",
    description="Document Intelligence and Retrieval-Augmented Generation API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "DocuMind API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }