"""Redis Stack HNSW Vector Semantic Cache Service."""

import logging
import time
import uuid
from typing import List, Optional, Dict, Any
import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import VectorField, TextField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = logging.getLogger(__name__)


class RedisSemanticCacheService:
    """High-performance semantic vector cache using Redis Stack HNSW Index."""

    def __init__(
        self,
        redis: Redis,
        index_name: str = "idx:zenith_cache",
        prefix: str = "cache:",
        dimension: int = 384,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
    ) -> None:
        self.redis = redis
        self.index_name = index_name
        self.prefix = prefix
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl

    async def init_index(self) -> None:
        """Create RediSearch HNSW vector index if it does not already exist."""
        if not self.redis:
            logger.warning("Redis client is not available. Skipping index initialization.")
            return

        try:
            # Check if index already exists
            try:
                await self.redis.ft(self.index_name).info()
                logger.info(f"RediSearch vector index '{self.index_name}' already exists.")
                return
            except Exception:
                logger.info(f"Creating RediSearch HNSW vector index '{self.index_name}'...")

            # Define schema
            schema = (
                VectorField(
                    "prompt_vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 10000,
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
                TextField("prompt"),
                TextField("response_text"),
                TagField("model"),
                NumericField("created_at"),
            )

            definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            await self.redis.ft(self.index_name).create_index(fields=schema, definition=definition)
            logger.info(f"Successfully created RediSearch index '{self.index_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize RediSearch index: {e}. Semantic caching will degrade gracefully.")

    async def get_similar_response(
        self,
        prompt_embedding: List[float],
        model: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Redis vector index for nearest neighbor and evaluate cosine similarity.

        In Redis Stack COSINE metric, distance d in [0, 2].
        Cosine similarity = 1 - d.
        """
        if not self.redis:
            return None

        target_threshold = threshold if threshold is not None else self.similarity_threshold

        try:
            query_vec_bytes = np.array(prompt_embedding, dtype=np.float32).tobytes()

            # RediSearch KNN Query
            query_str = "*=>[KNN 1 @prompt_vector $vec_param AS vector_score]"
            query = (
                Query(query_str)
                .sort_by("vector_score")
                .return_fields("response_text", "model", "created_at", "prompt", "vector_score")
                .dialect(2)
            )

            query_params = {"vec_param": query_vec_bytes}
            result = await self.redis.ft(self.index_name).search(query, query_params=query_params)

            if not result or result.total == 0 or not result.docs:
                return None

            best_match = result.docs[0]
            vector_score = float(getattr(best_match, "vector_score", 1.0))
            
            # For COSINE distance: similarity = 1 - distance
            similarity = 1.0 - vector_score

            logger.debug(
                f"Vector search match found with cosine distance: {vector_score:.4f}, similarity: {similarity:.4f} (threshold: {target_threshold})"
            )

            if similarity >= target_threshold:
                response_text = getattr(best_match, "response_text", "")
                cached_model = getattr(best_match, "model", model or "unknown")
                created_at = float(getattr(best_match, "created_at", time.time()))

                return {
                    "response_text": response_text,
                    "model": cached_model,
                    "similarity": round(similarity, 4),
                    "created_at": created_at,
                }

            return None
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}. Gracefully falling back to cache miss.")
            return None

    async def set_cache(
        self,
        prompt: str,
        prompt_embedding: List[float],
        response_text: str,
        model: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store prompt vector and response text in Redis Hash with TTL."""
        if not self.redis:
            return False

        cache_ttl = ttl or self.default_ttl
        cache_id = uuid.uuid4().hex
        key = f"{self.prefix}{cache_id}"

        try:
            vec_bytes = np.array(prompt_embedding, dtype=np.float32).tobytes()
            mapping = {
                "prompt": prompt,
                "response_text": response_text,
                "model": model,
                "created_at": time.time(),
                "prompt_vector": vec_bytes,
            }

            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, cache_ttl)
            await pipe.execute()
            logger.debug(f"Cached semantic entry '{key}' with TTL {cache_ttl}s.")
            return True
        except Exception as e:
            logger.warning(f"Failed to write entry to semantic cache: {e}.")
            return False
