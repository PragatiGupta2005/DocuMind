import os
from dotenv import load_dotenv

load_dotenv()
# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding Configuration
# Which embedding provider should DocuMind use?
# Supported:
#   local
#   api

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "local"
).lower().strip()


LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


API_EMBEDDING_MODEL = os.getenv(
    "API_EMBEDDING_MODEL",
    "text-embedding-3-small"
)


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)


EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32"
    )
)


EMBEDDING_NORMALIZE = os.getenv(
    "EMBEDDING_NORMALIZE",
    "true"
).lower() == "true"