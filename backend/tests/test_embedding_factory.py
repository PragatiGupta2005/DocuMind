from app.embeddings.embedding_factory import EmbeddingFactory


print("===================================")
print("Embedding Factory Test")
print("===================================")

# -----------------------------------
# Test 1: Local Provider
# -----------------------------------

print("\nTesting LOCAL provider...")
local_provider = EmbeddingFactory.get_provider("local")
print(
    "Provider Type:",
    type(local_provider).__name__
)

print(
    "Model:",
    local_provider.get_model_name()
)

# -----------------------------------
# Test 2: API Provider
# -----------------------------------

print("\nTesting API provider...")
api_provider = EmbeddingFactory.get_provider("api")
print(
    "Provider Type:",
    type(api_provider).__name__
)

print(
    "Model:",
    api_provider.get_model_name()
)


# -----------------------------------
# Test 3: Invalid Provider
# -----------------------------------
print("\nTesting INVALID provider...")
try:
    EmbeddingFactory.get_provider("unknown")
except ValueError as error:
    print("Expected error:")
    print(error)
    
print("\n===================================")
print("Factory Test Completed")
print("===================================")