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

    def get_dimensions(self) -> int:
        return 3


def test_embedding_model_name():
    embedding = TestEmbedding()

    assert embedding.get_model_name() == "test-model"


def test_embedding_dimensions():
    embedding = TestEmbedding()

    assert embedding.get_dimensions() == 3


def test_embedding_single():
    embedding = TestEmbedding()

    result = embedding.embed("Hello")

    assert result == [0.1, 0.2, 0.3]


def test_embedding_batch():
    embedding = TestEmbedding()

    result = embedding.embed_batch(
        ["Hello", "World"]
    )

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.1, 0.2, 0.3]