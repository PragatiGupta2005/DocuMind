from openai import OpenAI

from app.embeddings.base_embedding import BaseEmbedding

from app.core.settings import (
    API_EMBEDDING_MODEL,
    OPENAI_API_KEY
)


class APIEmbedding(BaseEmbedding):
    """
    API-based embedding provider using OpenAI.
    """

    def __init__(
        self,
        model_name: str = API_EMBEDDING_MODEL
    ):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.model_name = model_name

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
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

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )

        return response.data[0].embedding

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

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [
            item.embedding
            for item in response.data
        ]

    def get_model_name(self) -> str:
        """
        Return the embedding model name.
        """

        return self.model_name