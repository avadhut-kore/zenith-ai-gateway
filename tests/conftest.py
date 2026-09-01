"""Pytest fixtures and mock objects for asynchronous testing."""

import asyncio
from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.api.deps import (
    get_cache_service,
    get_embedding_service_dep,
    get_llm_client,
    get_rate_limiter,
    get_redis_client,
    get_settings,
)
from app.config import Settings
from app.core.rate_limiter import TokenBucketRateLimiter
from app.main import create_app
from app.services.cache_service import RedisSemanticCacheService
from app.services.embedding_service import EmbeddingService
from app.services.llm_client import LLMClient


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for asyncio testing."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing test configuration settings."""
    return Settings(
        APP_NAME="Zenith AI Gateway Test",
        DEBUG=True,
        SIMILARITY_THRESHOLD=0.95,
        CACHE_TTL_SECONDS=3600,
        RATE_LIMIT_TOKENS_PER_MIN=60000,
        OPENAI_API_KEY="test-openai-key",
    )


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock embedding service returning 384-dim normalized vector."""
    service = MagicMock(spec=EmbeddingService)
    # Return normalized 384-dimensional vector
    dummy_vec = [1.0 / np.sqrt(384)] * 384
    service.generate_embedding = AsyncMock(return_value=dummy_vec)
    service.embedding_to_bytes = staticmethod(lambda vec: np.array(vec, dtype=np.float32).tobytes())
    return service


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.script_load = AsyncMock(return_value="mock_script_sha")
    # Lua script evaluation mock: returns [allowed(1), remaining(59900), retry_after(0)]
    redis.evalsha = AsyncMock(return_value=[1, 59900, 0])
    return redis


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client returning predictable non-streaming and streaming responses."""
    client = MagicMock(spec=LLMClient)
    client.chat_completion = AsyncMock(return_value="Hello! I am Zenith AI Gateway.")

    async def mock_stream(messages, model, temperature=0.7, max_tokens=None) -> AsyncGenerator[str, None]:
        tokens = ["Hello", "!", " I", " am", " Zenith", " AI", " Gateway", "."]
        for token in tokens:
            yield token
            await asyncio.sleep(0.001)

    client.stream_chat_completion = mock_stream
    return client


@pytest.fixture
def mock_cache_service() -> MagicMock:
    """Mock semantic cache service."""
    cache = MagicMock(spec=RedisSemanticCacheService)
    cache.get_similar_response = AsyncMock(return_value=None)
    cache.set_cache = AsyncMock(return_value=True)
    cache.init_index = AsyncMock(return_value=None)
    return cache


@pytest.fixture
def mock_rate_limiter() -> MagicMock:
    """Mock rate limiter."""
    limiter = MagicMock(spec=TokenBucketRateLimiter)
    limiter.enforce_limit = AsyncMock(return_value=None)
    limiter.check_limit = AsyncMock(return_value=(True, 60000, 0))
    return limiter


@pytest.fixture
async def async_client(
    mock_settings: Settings,
    mock_redis: AsyncMock,
    mock_cache_service: MagicMock,
    mock_embedding_service: MagicMock,
    mock_llm_client: MagicMock,
    mock_rate_limiter: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with dependency overrides."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_settings] = lambda: mock_settings
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_cache_service] = lambda: mock_cache_service
    app.dependency_overrides[get_embedding_service_dep] = lambda: mock_embedding_service
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_rate_limiter] = lambda: mock_rate_limiter

    app.state.redis = mock_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
