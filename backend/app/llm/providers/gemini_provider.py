from app.llm.llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Gemini-based LLM provider.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        if not api_key:
            raise ValueError(
                "Gemini API key cannot be empty."
            )

        if not model_name:
            raise ValueError(
                "Gemini model name cannot be empty."
            )

        self.api_key = api_key
        self.model_name = model_name

        self._client = None

    def _get_client(self):
        """
        Lazily initialize the Gemini client.
        """

        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Google GenAI SDK is not installed."
            ) from exc

        self._client = genai.Client(
            api_key=self.api_key
        )

        return self._client

    def generate(
        self,
        prompt: str,
    ) -> str:

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        client = self._get_client()

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text

    def get_model_name(self) -> str:
        return self.model_name