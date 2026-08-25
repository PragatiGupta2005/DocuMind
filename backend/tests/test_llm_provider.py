import pytest

from app.llm.llm_factory import LLMFactory
from app.llm.llm_provider import LLMProvider
from app.llm.llm_service import LLMService


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        prompt: str,
    ) -> str:

        return f"Generated response for: {prompt}"

    def get_model_name(self) -> str:

        return "fake-model"


def test_fake_provider_generation():

    provider = FakeLLMProvider()

    result = provider.generate(
        "What is machine learning?"
    )

    assert (
        result
        == "Generated response for: What is machine learning?"
    )


def test_fake_provider_model_name():

    provider = FakeLLMProvider()

    assert (
        provider.get_model_name()
        == "fake-model"
    )


def test_llm_service_with_provider():

    provider = FakeLLMProvider()

    service = LLMService(
        provider=provider
    )

    result = service.generate(
        "Explain AI."
    )

    assert (
        result
        == "Generated response for: Explain AI."
    )


def test_llm_service_model_name():

    provider = FakeLLMProvider()

    service = LLMService(
        provider=provider
    )

    assert (
        service.get_model_name()
        == "fake-model"
    )


def test_llm_service_rejects_empty_prompt():

    provider = FakeLLMProvider()

    service = LLMService(
        provider=provider
    )

    with pytest.raises(ValueError):

        service.generate("")


def test_factory_rejects_unknown_provider():

    with pytest.raises(ValueError):

        LLMFactory.create_provider(
            provider_name="unknown"
        )


def test_factory_requires_gemini_api_key():

    with pytest.raises(ValueError):

        LLMFactory.create_provider(
            provider_name="gemini",
            model_name="test-model",
        )


def test_factory_requires_gemini_model():

    with pytest.raises(ValueError):

        LLMFactory.create_provider(
            provider_name="gemini",
            api_key="fake-key",
        )


def test_factory_creates_gemini_provider():

    provider = LLMFactory.create_provider(
        provider_name="gemini",
        api_key="fake-key",
        model_name="test-model",
    )

    assert provider.get_model_name() == "test-model"