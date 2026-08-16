from app.embeddings.base_embedding import BaseEmbedding
from app.embeddings.local_embedding import LocalEmbedding
from app.embeddings.api_embedding import APIEmbedding

from app.core.settings import EMBEDDING_PROVIDER


class EmbeddingFactory:
    """
    Factory responsible for creating the configured
    embedding provider.
    """

    _providers = {
        "local": LocalEmbedding,
        "api": APIEmbedding
    }

    @classmethod
    def get_provider(
        cls,
        provider: str = EMBEDDING_PROVIDER
    ) -> BaseEmbedding:
        """
        Return an embedding provider based on the
        configured provider name.
        """

        provider = provider.lower().strip()

        provider_class = cls._providers.get(provider)

        if provider_class is None:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Available providers: "
                f"{', '.join(cls._providers.keys())}"
            )

        return provider_class()