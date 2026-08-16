from app.embeddings.embedding_factory import EmbeddingFactory


try:

    provider = EmbeddingFactory.get_provider(
        "unknown"
    )

except ValueError as error:

    print("Expected error:")
    print(error)