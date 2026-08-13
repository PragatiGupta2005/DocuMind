from app.embeddings.base_embedding import BaseEmbedding


class TestEmbedding(BaseEmbedding):

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def get_model_name(self) -> str:
        return "test-model"


embedding = TestEmbedding()

print("Model:", embedding.get_model_name())

print(
    "Single:",
    embedding.embed("Hello")
)

print(
    "Batch:",
    embedding.embed_batch(
        ["Hello", "World"]
    )
)