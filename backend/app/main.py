from fastapi import FastAPI


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