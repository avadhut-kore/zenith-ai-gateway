"""Unit and integration tests for Redis HNSW semantic caching."""

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from app.services.cache_service import RedisSemanticCacheService
from app.services.embedding_service import EmbeddingService


@pytest.mark.asyncio
async def test_embedding_service_byte_conversion():
    """Verify conversion of float list embedding to 32-bit binary buffer."""
    vec = [0.1, 0.2, 0.3, 0.4]
    byte_repr = EmbeddingService.embedding_to_bytes(vec)
    reconstructed = np.frombuffer(byte_repr, dtype=np.float32).tolist()
    assert len(reconstructed) == len(vec)
    for orig, rec in zip(vec, reconstructed):
        assert pytest.approx(orig, rel=1e-5) == rec


@pytest.mark.asyncio
async def test_cache_hit_when_similarity_above_threshold():
    """Verify cache hit when RediSearch returns cosine distance <= 0.05 (similarity >= 0.95)."""
    mock_redis = AsyncMock()
    mock_ft = MagicMock()
    mock_redis.ft.return_value = mock_ft

    # Mock RediSearch doc response: vector_score = 0.03 -> similarity = 0.97
    mock_doc = MagicMock()
    mock_doc.vector_score = "0.03"
    mock_doc.response_text = "Cached AI completion."
    mock_doc.model = "gpt-4o"
    mock_doc.created_at = 1700000000.0

    mock_search_result = MagicMock()
    mock_search_result.total = 1
    mock_search_result.docs = [mock_doc]
    mock_ft.search = AsyncMock(return_value=mock_search_result)

    cache_service = RedisSemanticCacheService(
        redis=mock_redis,
        similarity_threshold=0.95,
    )

    query_vec = [0.1] * 384
    result = await cache_service.get_similar_response(query_vec)

    assert result is not None
    assert result["response_text"] == "Cached AI completion."
    assert result["model"] == "gpt-4o"
    assert result["similarity"] == pytest.approx(0.97, rel=1e-3)


@pytest.mark.asyncio
async def test_cache_miss_when_similarity_below_threshold():
    """Verify cache miss when RediSearch returns cosine distance > 0.05 (similarity < 0.95)."""
    mock_redis = AsyncMock()
    mock_ft = MagicMock()
    mock_redis.ft.return_value = mock_ft

    # Mock RediSearch doc response: vector_score = 0.15 -> similarity = 0.85
    mock_doc = MagicMock()
    mock_doc.vector_score = "0.15"
    mock_doc.response_text = "Unrelated cached completion."
    mock_doc.model = "gpt-4o"
    mock_doc.created_at = 1700000000.0

    mock_search_result = MagicMock()
    mock_search_result.total = 1
    mock_search_result.docs = [mock_doc]
    mock_ft.search = AsyncMock(return_value=mock_search_result)

    cache_service = RedisSemanticCacheService(
        redis=mock_redis,
        similarity_threshold=0.95,
    )

    query_vec = [0.1] * 384
    result = await cache_service.get_similar_response(query_vec)

    assert result is None


@pytest.mark.asyncio
async def test_cache_miss_on_empty_index():
    """Verify cache miss when no matches are found in Redis."""
    mock_redis = AsyncMock()
    mock_ft = MagicMock()
    mock_redis.ft.return_value = mock_ft

    mock_search_result = MagicMock()
    mock_search_result.total = 0
    mock_search_result.docs = []
    mock_ft.search = AsyncMock(return_value=mock_search_result)

    cache_service = RedisSemanticCacheService(redis=mock_redis)
    result = await cache_service.get_similar_response([0.1] * 384)
    assert result is None


@pytest.mark.asyncio
async def test_set_cache():
    """Verify writing entry to Redis with pipeline and TTL."""
    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_redis.pipeline.return_value = mock_pipe
    mock_pipe.execute = AsyncMock(return_value=[True, True])

    cache_service = RedisSemanticCacheService(redis=mock_redis, default_ttl=1800)
    success = await cache_service.set_cache(
        prompt="Explain quantum entanglement",
        prompt_embedding=[0.05] * 384,
        response_text="Quantum entanglement is a physical phenomenon...",
        model="gpt-4o",
        ttl=1800,
    )

    assert success is True
    mock_pipe.hset.assert_called_once()
    mock_pipe.expire.assert_called_once()


@pytest.mark.asyncio
async def test_graceful_degradation_when_redis_unavailable():
    """Verify cache service degrades gracefully when Redis client is None."""
    cache_service = RedisSemanticCacheService(redis=None)
    result = await cache_service.get_similar_response([0.1] * 384)
    assert result is None

    success = await cache_service.set_cache(
        prompt="test",
        prompt_embedding=[0.1] * 384,
        response_text="test",
        model="test",
    )
    assert success is False
