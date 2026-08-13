from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Abstract base class for all embedding providers.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text.
        """
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the embedding model.
        """
        pass