from app.core.settings import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
)

from app.llm.llm_factory import LLMFactory
from app.llm.llm_provider import LLMProvider


class LLMService:
    """
    Service layer responsible for text generation.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
    ):

        if provider is not None:

            self.provider = provider

        else:

            self.provider = (
                LLMFactory.create_provider(
                    provider_name=LLM_PROVIDER,
                    api_key=GEMINI_API_KEY,
                    model_name=LLM_MODEL,
                )
            )

    def generate(
        self,
        prompt: str,
    ) -> str:

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        return self.provider.generate(
            prompt
        )

    def get_model_name(self) -> str:

        return self.provider.get_model_name()