from app.llm.llm_factory import LLMFactory
from app.llm.llm_provider import LLMProvider


class LLMService:
    """
    Service layer responsible for text generation.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        provider_name: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ):

        if provider is not None:

            self.provider = provider

        else:

            if provider_name is None:
                raise ValueError(
                    "LLM provider name is required."
                )

            self.provider = (
                LLMFactory.create_provider(
                    provider_name=provider_name,
                    api_key=api_key,
                    model_name=model_name,
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