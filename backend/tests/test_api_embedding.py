from unittest.mock import MagicMock

from app.embeddings.api_embedding import APIEmbedding


def test_api_embedding_model_name():
    embedding = APIEmbedding.__new__(
        APIEmbedding
    )

    embedding.model_name = "gemini-embedding-001"

    assert (
        embedding.get_model_name()
        == "gemini-embedding-001"
    )


def test_api_embedding_single(monkeypatch):
    embedding = APIEmbedding.__new__(
        APIEmbedding
    )

    embedding.model_name = "gemini-embedding-001"

    fake_vector = [0.1] * 3072

    fake_response = MagicMock()
    fake_response.embeddings = [
        MagicMock(values=fake_vector)
    ]

    embedding.client = MagicMock()

    embedding.client.models.embed_content.return_value = (
        fake_response
    )

    vector = embedding.embed(
        "Machine Learning is a subset of Artificial Intelligence."
    )

    assert isinstance(vector, list)

    assert len(vector) == 3072

    embedding.client.models.embed_content.assert_called_once()


def test_api_embedding_batch(monkeypatch):
    embedding = APIEmbedding.__new__(
        APIEmbedding
    )

    embedding.model_name = "gemini-embedding-001"

    texts = [
        "Artificial Intelligence is a field of computer science.",
        "Machine Learning allows systems to learn from data.",
        "Deep Learning uses neural networks.",
    ]

    fake_response = MagicMock()

    fake_response.embeddings = [
        MagicMock(values=[0.1] * 3072),
        MagicMock(values=[0.2] * 3072),
        MagicMock(values=[0.3] * 3072),
    ]

    embedding.client = MagicMock()

    embedding.client.models.embed_content.return_value = (
        fake_response
    )

    vectors = embedding.embed_batch(texts)

    assert isinstance(vectors, list)

    assert len(vectors) == len(texts)

    assert all(
        len(vector) == 3072
        for vector in vectors
    )

    embedding.client.models.embed_content.assert_called_once()