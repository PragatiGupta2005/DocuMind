from google import genai
from google.genai import types

from app.embeddings.base_embedding import BaseEmbedding

from app.core.settings import (
    API_EMBEDDING_MODEL,
    GEMINI_API_KEY
)


class APIEmbedding(BaseEmbedding):
    """
    API-based embedding provider using Gemini.
    """

    def __init__(
        self,
        model_name: str = API_EMBEDDING_MODEL
    ):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
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

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )

        return response.embeddings[0].values

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

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def get_model_name(self) -> str:
        """
        Return the embedding model name.
        """

        return self.model_name