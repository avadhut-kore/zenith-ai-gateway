"""Zenith AI Gateway main application entry point and lifespan lifecycle."""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis, ConnectionPool

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.core.telemetry import setup_telemetry
from app.services.cache_service import RedisSemanticCacheService
from app.services.embedding_service import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("zenith.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager handling startup initialization and graceful shutdown."""
    settings = get_settings()

    # 1. Initialize OpenTelemetry Tracing
    setup_telemetry(
        service_name=settings.OTEL_SERVICE_NAME,
        otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
    )
    logger.info(f"Initialized telemetry for '{settings.OTEL_SERVICE_NAME}'.")

    # 2. Initialize Redis Connection Pool
    redis_pool = None
    redis_client = None
    try:
        redis_pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=50,
            decode_responses=False,  # Keep binary format for vector embeddings
        )
        redis_client = Redis(connection_pool=redis_pool)
        await redis_client.ping()
        logger.info(f"Successfully connected to Redis Stack at {settings.REDIS_URL}.")
        app.state.redis = redis_client

        # 3. Ensure RediSearch HNSW vector index exists
        cache_service = RedisSemanticCacheService(
            redis=redis_client,
            index_name=settings.CACHE_INDEX_NAME,
            prefix=settings.CACHE_KEY_PREFIX,
            dimension=settings.EMBEDDING_DIMENSION,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            default_ttl=settings.CACHE_TTL_SECONDS,
        )
        await cache_service.init_index()
    except Exception as e:
        logger.warning(
            f"Could not connect to Redis Stack or initialize index: {e}. Gateway will run in cache-degraded mode."
        )
        app.state.redis = None

    # 4. Preload Sentence Transformers Embedding Model in background thread
    logger.info(f"Pre-warming embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
    embedding_service = EmbeddingService.get_instance(model_name=settings.EMBEDDING_MODEL_NAME)
    asyncio.create_task(embedding_service.generate_embedding("Zenith AI Gateway warm-up prompt"))

    logger.info(f"🚀 {settings.APP_NAME} v{__version__} startup complete and ready to accept traffic.")

    yield

    # Graceful Shutdown
    logger.info("Initiating graceful shutdown of Zenith AI Gateway...")
    if redis_client:
        await redis_client.aclose()
    if redis_pool:
        await redis_pool.disconnect()
    logger.info("Shutdown completed cleanly.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description="High-throughput, low-latency asynchronous LLM reverse proxy with Redis HNSW semantic vector caching.",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error occurred within Zenith Gateway.",
                    "type": "internal_error",
                    "code": 500,
                }
            },
        )

    # Include Routes
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        loop="uvloop",
    )
