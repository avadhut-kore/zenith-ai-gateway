"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Zenith AI Gateway Settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General App Settings
    APP_NAME: str = "Zenith AI Gateway"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Redis Stack (Vector Search)
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM Provider API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Semantic Cache Configuration
    SIMILARITY_THRESHOLD: float = 0.95
    CACHE_TTL_SECONDS: int = 3600
    CACHE_INDEX_NAME: str = "idx:zenith_cache"
    CACHE_KEY_PREFIX: str = "cache:"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Rate Limiting Configuration
    RATE_LIMIT_TOKENS_PER_MIN: int = 60000

    # Observability & Tracing
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_SERVICE_NAME: str = "zenith-ai-gateway"
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
