from sentence_transformers import SentenceTransformer

from app.embeddings.base_embedding import BaseEmbedding

from app.core.settings import (
    LOCAL_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_NORMALIZE
)


class LocalEmbedding(BaseEmbedding):
    """
    Local embedding provider using Sentence Transformers.
    """

    def __init__(
        self,
        model_name: str = LOCAL_EMBEDDING_MODEL
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed(
        self,
        text: str
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=EMBEDDING_NORMALIZE
        )

        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return []

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Texts cannot contain empty values."
            )

        vectors = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=EMBEDDING_NORMALIZE
        )

        return vectors.tolist()

    def get_model_name(self) -> str:
        """
        Return the embedding model name.
        """

        return self.model_name

    def get_dimensions(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """

        return 384