from app.embeddings.embedding_factory import EmbeddingFactory


print("===================================")
print("Factory Configuration Test")
print("===================================")


provider = EmbeddingFactory.get_provider()


print(
    "Provider Type:",
    type(provider).__name__
)

print(
    "Model:",
    provider.get_model_name()
)


print("===================================")