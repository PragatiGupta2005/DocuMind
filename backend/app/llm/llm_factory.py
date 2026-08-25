from app.llm.llm_provider import LLMProvider
from app.llm.providers.gemini_provider import GeminiProvider


class LLMFactory:
    """
    Creates the configured LLM provider.
    """

    @staticmethod
    def create_provider(
        provider_name: str,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> LLMProvider:

        provider = provider_name.lower().strip()

        if provider == "gemini":

            if not api_key:
                raise ValueError(
                    "Gemini API key is required."
                )

            if not model_name:
                raise ValueError(
                    "Gemini model name is required."
                )

            return GeminiProvider(
                api_key=api_key,
                model_name=model_name,
            )

        raise ValueError(
            f"Unsupported LLM provider: {provider_name}"
        )