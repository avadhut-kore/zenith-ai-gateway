"""Redis-based Token Bucket Rate Limiter per tenant/API key."""

import logging
import math
import time
from typing import Optional, Tuple
from fastapi import HTTPException, status
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Lua script to atomically refill and consume tokens from the bucket
TOKEN_BUCKET_LUA_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local fill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if tokens == nil then
    tokens = capacity
    last_updated = now
else
    local elapsed = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + (elapsed * fill_rate))
    last_updated = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {1, math.floor(tokens), 0}
else
    local deficit = requested - tokens
    local retry_after = math.ceil(deficit / fill_rate)
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {0, math.floor(tokens), retry_after}
end
"""


class TokenBucketRateLimiter:
    """Async Redis token bucket rate limiter tracking consumed tokens per tenant."""

    def __init__(self, default_tokens_per_min: int = 60000) -> None:
        self.default_tokens_per_min = default_tokens_per_min
        self._script_sha: Optional[str] = None

    async def _get_script_sha(self, redis: Redis) -> str:
        """Register and return cached SHA for Lua script."""
        if not self._script_sha:
            self._script_sha = await redis.script_load(TOKEN_BUCKET_LUA_SCRIPT)
        return self._script_sha

    async def check_limit(
        self,
        redis: Optional[Redis],
        tenant_id: str,
        requested_tokens: int = 100,
        tokens_per_min: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Check if tenant has sufficient tokens in their bucket.

        Returns:
            Tuple of (is_allowed, remaining_tokens, retry_after_seconds)
        """
        if redis is None:
            # Graceful degradation if Redis is unavailable
            return True, self.default_tokens_per_min, 0

        capacity = tokens_per_min or self.default_tokens_per_min
        fill_rate = capacity / 60.0  # tokens per second
        now = time.time()
        key = f"ratelimit:bucket:{tenant_id}"
        ttl = 300  # 5 minutes idle TTL

        try:
            sha = await self._get_script_sha(redis)
            result = await redis.evalsha(
                sha,
                1,
                key,
                str(capacity),
                str(fill_rate),
                str(requested_tokens),
                str(now),
                str(ttl),
            )
            is_allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = int(result[2])
            return is_allowed, remaining, retry_after
        except Exception as e:
            logger.error(f"Rate limiting evaluation failed: {e}. Degrading gracefully to allow request.")
            return True, capacity, 0

    async def enforce_limit(
        self,
        redis: Optional[Redis],
        tenant_id: str,
        requested_tokens: int = 100,
        tokens_per_min: Optional[int] = None,
    ) -> None:
        """Enforce rate limit, raising HTTP 429 if quota exceeded."""
        is_allowed, remaining, retry_after = await self.check_limit(
            redis=redis,
            tenant_id=tenant_id,
            requested_tokens=requested_tokens,
            tokens_per_min=tokens_per_min,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for tenant '{tenant_id}'. Remaining tokens: {remaining}.",
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )
