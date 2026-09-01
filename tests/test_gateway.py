"""End-to-end integration and API tests for Zenith AI Gateway."""

from unittest.mock import AsyncMock
from fastapi import HTTPException, status
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_healthz_endpoint(async_client: AsyncClient):
    """Test /healthz endpoint returns healthy status and component states."""
    response = await async_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["redis"] == "connected"
    assert "embedding_model" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client: AsyncClient):
    """Test /metrics Prometheus endpoint returns metrics payload."""
    response = await async_client.get("/metrics")
    assert response.status_code == 200
    assert "zenith_requests_total" in response.text


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_cache_miss(
    async_client: AsyncClient,
    mock_cache_service,
    mock_llm_client,
):
    """Test standard non-streaming completion resulting in a cache miss and LLM forward."""
    mock_cache_service.get_similar_response.return_value = None

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the speed of light?"}
        ],
        "temperature": 0.5,
        "stream": False,
        "tenant_id": "test-tenant",
    }

    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers.get("x-cache") == "MISS"

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o"
    assert body["cache_hit"] is False
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["content"] == "Hello! I am Zenith AI Gateway."

    mock_llm_client.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_cache_hit(
    async_client: AsyncClient,
    mock_cache_service,
    mock_llm_client,
):
    """Test standard non-streaming completion returning cached response with X-Cache: HIT."""
    mock_cache_service.get_similar_response.return_value = {
        "response_text": "The speed of light in vacuum is approximately 299,792,458 m/s.",
        "model": "gpt-4o",
        "similarity": 0.982,
        "created_at": 1700000000.0,
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the exact speed of light in vacuum?"}
        ],
        "stream": False,
        "tenant_id": "test-tenant",
    }

    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers.get("x-cache") == "HIT"
    assert response.headers.get("x-cache-similarity") == "0.982"

    body = response.json()
    assert body["cache_hit"] is True
    assert "299,792,458" in body["choices"][0]["message"]["content"]

    # LLM should not be called on cache hit
    mock_llm_client.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_chat_completions_streaming_cache_miss(
    async_client: AsyncClient,
    mock_cache_service,
):
    """Test Server-Sent Events (SSE) streaming completion on cache miss."""
    mock_cache_service.get_similar_response.return_value = None

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Stream a response to me"}
        ],
        "stream": True,
        "tenant_id": "test-tenant",
    }

    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers.get("x-cache") == "MISS"
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Collect SSE chunks
    content = response.text
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    assert len(lines) > 0
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_chat_completions_streaming_cache_hit(
    async_client: AsyncClient,
    mock_cache_service,
):
    """Test Server-Sent Events (SSE) streaming completion on cache hit."""
    mock_cache_service.get_similar_response.return_value = {
        "response_text": "This is a cached streaming test.",
        "model": "gpt-4o",
        "similarity": 0.99,
        "created_at": 1700000000.0,
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "This is a cached streaming query"}
        ],
        "stream": True,
        "tenant_id": "test-tenant",
    }

    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers.get("x-cache") == "HIT"
    assert "text/event-stream" in response.headers.get("content-type", "")

    content = response.text
    lines = [line for line in content.split("\n") if line.startswith("data: ")]
    assert len(lines) > 0
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_rate_limiter_exceeded_429(
    async_client: AsyncClient,
    mock_rate_limiter,
):
    """Test that HTTP 429 is returned when rate limit is exceeded."""
    mock_rate_limiter.enforce_limit.side_effect = HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded for tenant 'test-tenant'.",
        headers={"Retry-After": "30"},
    )

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "tenant_id": "test-tenant",
    }

    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text
    assert response.headers.get("retry-after") == "30"
