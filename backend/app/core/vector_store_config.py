from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class VectorStoreSettings(BaseSettings):
    """
    Configuration for the vector store.
    """

    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


vector_store_settings = VectorStoreSettings()