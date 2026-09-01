"""FastAPI dependency injection providers."""

from typing import Optional
from fastapi import Depends, Request
from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.core.rate_limiter import TokenBucketRateLimiter
from app.services.cache_service import RedisSemanticCacheService
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_client import LLMClient


def get_redis_client(request: Request) -> Optional[Redis]:
    """Retrieve async Redis client instance from application state."""
    return getattr(request.app.state, "redis", None)


def get_cache_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RedisSemanticCacheService:
    """Retrieve or construct RedisSemanticCacheService."""
    redis = get_redis_client(request)
    return RedisSemanticCacheService(
        redis=redis,
        index_name=settings.CACHE_INDEX_NAME,
        prefix=settings.CACHE_KEY_PREFIX,
        dimension=settings.EMBEDDING_DIMENSION,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        default_ttl=settings.CACHE_TTL_SECONDS,
    )


def get_embedding_service_dep(
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Retrieve singleton EmbeddingService."""
    return get_embedding_service(model_name=settings.EMBEDDING_MODEL_NAME)


def get_llm_client(
    settings: Settings = Depends(get_settings),
) -> LLMClient:
    """Retrieve LLMClient configured with API keys."""
    return LLMClient(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
    )


def get_rate_limiter(
    settings: Settings = Depends(get_settings),
) -> TokenBucketRateLimiter:
    """Retrieve TokenBucketRateLimiter configured with default rate limits."""
    return TokenBucketRateLimiter(default_tokens_per_min=settings.RATE_LIMIT_TOKENS_PER_MIN)
