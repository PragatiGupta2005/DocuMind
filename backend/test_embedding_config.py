from app.core.settings import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    API_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_NORMALIZE
)


print("================================")
print("Embedding Configuration")
print("================================")

print("Provider        :", EMBEDDING_PROVIDER)

print("Local Model     :", LOCAL_EMBEDDING_MODEL)

print("API Model       :", API_EMBEDDING_MODEL)

print("Batch Size      :", EMBEDDING_BATCH_SIZE)

print("Normalize       :", EMBEDDING_NORMALIZE)

print("================================")