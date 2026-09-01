from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def embed(
        self,
        text: str
    ) -> list[float]:
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Return the dimensionality of the
        generated embeddings.
        """
        pass