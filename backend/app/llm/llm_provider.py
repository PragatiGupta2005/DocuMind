from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text from a prompt.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the underlying model.
        """
        raise NotImplementedError