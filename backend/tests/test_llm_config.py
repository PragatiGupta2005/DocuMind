from app.core.settings import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
)


def test_llm_provider_is_configured():

    assert LLM_PROVIDER == "gemini"


def test_llm_model_is_configured():

    assert LLM_MODEL


def test_gemini_api_key_setting_exists():

    assert isinstance(
        GEMINI_API_KEY,
        str,
    )