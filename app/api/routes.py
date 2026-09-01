"""API route handlers for chat completions, health status, and metrics."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from app import __version__
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
from app.core.telemetry import (
    ACTIVE_REQUESTS,
    CACHE_LATENCY,
    REQUEST_COUNT,
    TOKEN_COUNT,
    TTFT_HISTOGRAM,
    get_tracer,
)
from app.schemas.gateway import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    HealthResponse,
    StreamChoice,
    StreamDelta,
    UsageInfo,
)
from app.services.cache_service import RedisSemanticCacheService
from app.services.embedding_service import EmbeddingService
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter()
tracer = get_tracer("zenith.routes")


@router.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check(
    redis: Optional[Redis] = Depends(get_redis_client),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """System health check verifying Redis connection and embedding model status."""
    redis_status = "disconnected"
    if redis:
        try:
            pong = await redis.ping()
            if pong:
                redis_status = "connected"
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            redis_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if redis_status == "connected" else "degraded",
        redis=redis_status,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        version=__version__,
    )


@router.get("/metrics", tags=["Observability"])
async def get_metrics() -> Response:
    """Export Prometheus metrics in standard exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post(
    "/v1/chat/completions",
    response_model=None,
    tags=["Chat Completions"],
    responses={
        200: {
            "description": "Successful chat completion or Server-Sent Events token stream.",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        },
        429: {"description": "Rate limit exceeded."},
    },
)
async def chat_completions(
    request_body: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    redis: Optional[Redis] = Depends(get_redis_client),
    cache_service: RedisSemanticCacheService = Depends(get_cache_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service_dep),
    llm_client: LLMClient = Depends(get_llm_client),
    rate_limiter: TokenBucketRateLimiter = Depends(get_rate_limiter),
    settings: Settings = Depends(get_settings),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
):
    """OpenAI-compatible chat completion proxy endpoint with semantic caching, streaming, and rate limiting."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    tenant_id = x_tenant_id or request_body.tenant_id or "default"

    try:
        with tracer.start_as_current_span("chat_completions") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("model", request_body.model)
            span.set_attribute("stream", request_body.stream)

            # 1. Enforce Token Bucket Rate Limiting
            estimated_tokens = max(len(request_body.extract_prompt_text()) // 4, 50)
            with tracer.start_as_current_span("rate_limit_check"):
                await rate_limiter.enforce_limit(
                    redis=redis,
                    tenant_id=tenant_id,
                    requested_tokens=estimated_tokens,
                )

            # 2. Compute Dense Vector Embedding for the incoming prompt
            canonical_prompt = request_body.extract_prompt_text()
            with tracer.start_as_current_span("generate_embedding"):
                prompt_embedding = await embedding_service.generate_embedding(canonical_prompt)

            # 3. Check Semantic Cache (unless explicitly skipped)
            cached_result = None
            if not request_body.skip_cache:
                with tracer.start_as_current_span("cache_lookup") as cache_span:
                    cache_start = time.time()
                    cached_result = await cache_service.get_similar_response(
                        prompt_embedding=prompt_embedding,
                        model=request_body.model,
                        threshold=settings.SIMILARITY_THRESHOLD,
                    )
                    cache_latency = time.time() - cache_start
                    CACHE_LATENCY.observe(cache_latency)
                    cache_span.set_attribute("cache_latency_ms", cache_latency * 1000)
                    cache_span.set_attribute("cache_hit", cached_result is not None)

            # ==========================================
            # CASE A: CACHE HIT
            # ==========================================
            if cached_result:
                logger.info(
                    f"Semantic CACHE HIT (similarity: {cached_result['similarity']:.4f}) for model '{request_body.model}'."
                )
                REQUEST_COUNT.labels(model=request_body.model, cache_status="hit", status_code="200").inc()
                cached_text = cached_result["response_text"]
                completion_tokens = len(cached_text.split())

                if not request_body.stream:
                    response.headers["X-Cache"] = "HIT"
                    response.headers["X-Cache-Similarity"] = str(cached_result["similarity"])
                    return ChatCompletionResponse(
                        model=request_body.model,
                        choices=[
                            ChatChoice(
                                index=0,
                                message=ChatMessage(role="assistant", content=cached_text),
                                finish_reason="stop",
                            )
                        ],
                        usage=UsageInfo(
                            prompt_tokens=estimated_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=estimated_tokens + completion_tokens,
                        ),
                        cache_hit=True,
                    )
                else:
                    # Stream cached content as simulated SSE chunks
                    async def stream_cached_response() -> AsyncGenerator[str, None]:
                        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                        # Emit initial role delta
                        init_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            model=request_body.model,
                            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant"))],
                        )
                        yield f"data: {init_chunk.model_dump_json()}\n\n"

                        # Split text into small token batches for smooth streaming
                        words = cached_text.split(" ")
                        for i, word in enumerate(words):
                            token = word if i == len(words) - 1 else word + " "
                            chunk = ChatCompletionChunk(
                                id=chunk_id,
                                model=request_body.model,
                                choices=[StreamChoice(index=0, delta=StreamDelta(content=token))],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            await asyncio.sleep(0.005)

                        # Emit finish chunk
                        done_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            model=request_body.model,
                            choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")],
                        )
                        yield f"data: {done_chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"

                    headers = {
                        "X-Cache": "HIT",
                        "X-Cache-Similarity": str(cached_result["similarity"]),
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                    return StreamingResponse(stream_cached_response(), media_type="text/event-stream", headers=headers)

            # ==========================================
            # CASE B: CACHE MISS -> FORWARD TO LLM
            # ==========================================
            logger.info(f"Semantic CACHE MISS for model '{request_body.model}'. Routing to downstream LLM provider.")
            REQUEST_COUNT.labels(model=request_body.model, cache_status="miss", status_code="200").inc()
            messages_payload = [msg.model_dump(exclude_none=True) for msg in request_body.messages]

            if not request_body.stream:
                with tracer.start_as_current_span("llm_completion"):
                    response_text = await llm_client.chat_completion(
                        messages=messages_payload,
                        model=request_body.model,
                        temperature=request_body.temperature or 0.7,
                        max_tokens=request_body.max_tokens,
                    )

                completion_tokens = len(response_text.split())
                TOKEN_COUNT.labels(model=request_body.model, token_type="prompt").inc(estimated_tokens)
                TOKEN_COUNT.labels(model=request_body.model, token_type="completion").inc(completion_tokens)

                # Asynchronously cache the generated completion in Redis
                background_tasks.add_task(
                    cache_service.set_cache,
                    prompt=canonical_prompt,
                    prompt_embedding=prompt_embedding,
                    response_text=response_text,
                    model=request_body.model,
                    ttl=settings.CACHE_TTL_SECONDS,
                )

                response.headers["X-Cache"] = "MISS"
                return ChatCompletionResponse(
                    model=request_body.model,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=response_text),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=estimated_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=estimated_tokens + completion_tokens,
                    ),
                    cache_hit=False,
                )
            else:
                # Streaming Response over Server-Sent Events (SSE)
                async def event_generator() -> AsyncGenerator[str, None]:
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    accumulated_response: list[str] = []
                    first_token_emitted = False
                    stream_start = time.time()

                    try:
                        # Initial role announcement
                        init_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            model=request_body.model,
                            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant"))],
                        )
                        yield f"data: {init_chunk.model_dump_json()}\n\n"

                        async for token in llm_client.stream_chat_completion(
                            messages=messages_payload,
                            model=request_body.model,
                            temperature=request_body.temperature or 0.7,
                            max_tokens=request_body.max_tokens,
                        ):
                            if not first_token_emitted:
                                ttft = time.time() - stream_start
                                TTFT_HISTOGRAM.observe(ttft)
                                first_token_emitted = True

                            accumulated_response.append(token)
                            chunk = ChatCompletionChunk(
                                id=chunk_id,
                                model=request_body.model,
                                choices=[StreamChoice(index=0, delta=StreamDelta(content=token))],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"

                        # Final stop chunk
                        stop_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            model=request_body.model,
                            choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")],
                        )
                        yield f"data: {stop_chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"

                        # Complete text aggregation and asynchronous cache update
                        full_response_text = "".join(accumulated_response)
                        if full_response_text:
                            TOKEN_COUNT.labels(model=request_body.model, token_type="completion").inc(
                                len(full_response_text.split())
                            )
                            await cache_service.set_cache(
                                prompt=canonical_prompt,
                                prompt_embedding=prompt_embedding,
                                response_text=full_response_text,
                                model=request_body.model,
                                ttl=settings.CACHE_TTL_SECONDS,
                            )
                    except Exception as err:
                        logger.error(f"Error during stream generation: {err}")
                        error_payload = json.dumps({"error": {"message": str(err), "type": "gateway_stream_error"}})
                        yield f"data: {error_payload}\n\n"

                headers = {
                    "X-Cache": "MISS",
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
                return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

    finally:
        ACTIVE_REQUESTS.dec()
