"""Redis storage backend for distributed circuit breakers."""

import json
from typing import Any

from cbreaker.storage.base import BaseStorage


class RedisStorage(BaseStorage):
    """
    Redis storage backend.

    Stores circuit breaker state in Redis, enabling shared state across
    multiple application instances (horizontal scaling).

    Supports both sync (redis-py) and async (redis.asyncio) clients.

    Example (sync):
        import redis
        redis_client = redis.Redis(host='localhost', port=6379)
        storage = RedisStorage(sync_client=redis_client)
        breaker = CircuitBreaker(name="my_service", storage=storage)

    Example (async):
        import redis.asyncio as aioredis
        redis_client = aioredis.Redis(host='localhost', port=6379)
        storage = RedisStorage(async_client=redis_client)
        breaker = CircuitBreaker(name="my_service", storage=storage)
    """

    def __init__(
        self,
        sync_client: Any | None = None,
        async_client: Any | None = None,
        key_prefix: str = "cbreaker:",
        default_ttl: int | None = 86400,  # 24 hours default
    ):
        """
        Initialize the Redis storage.

        Args:
            sync_client: Synchronous Redis client (redis-py).
            async_client: Asynchronous Redis client (redis.asyncio).
            key_prefix: Prefix for all keys stored in Redis.
            default_ttl: Default TTL in seconds for stored keys.
        """
        if sync_client is None and async_client is None:
            raise ValueError("At least one of sync_client or async_client is required")

        self._sync_client = sync_client
        self._async_client = async_client
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """Create the full Redis key with prefix."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, state: dict[str, Any]) -> str:
        """Serialize state to JSON string."""
        return json.dumps(state)

    def _deserialize(self, data: str | bytes | None) -> dict[str, Any] | None:
        """Deserialize JSON string to state dict."""
        if data is None:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)

    def get(self, key: str) -> dict[str, Any] | None:
        """Get the state for a circuit breaker (sync)."""
        if self._sync_client is None:
            raise RuntimeError("Sync client not configured. Use get_async instead.")
        data = self._sync_client.get(self._make_key(key))
        return self._deserialize(data)

    def set(self, key: str, state: dict[str, Any], ttl: int | None = None) -> None:
        """Set the state for a circuit breaker (sync)."""
        if self._sync_client is None:
            raise RuntimeError("Sync client not configured. Use set_async instead.")
        full_key = self._make_key(key)
        serialized = self._serialize(state)
        effective_ttl = ttl if ttl is not None else self._default_ttl

        if effective_ttl is not None:
            self._sync_client.setex(full_key, effective_ttl, serialized)
        else:
            self._sync_client.set(full_key, serialized)

    def delete(self, key: str) -> None:
        """Delete the state for a circuit breaker (sync)."""
        if self._sync_client is None:
            raise RuntimeError("Sync client not configured. Use delete_async instead.")
        self._sync_client.delete(self._make_key(key))

    async def get_async(self, key: str) -> dict[str, Any] | None:
        """Get the state for a circuit breaker (async)."""
        if self._async_client is None:
            raise RuntimeError("Async client not configured. Use get instead.")
        data = await self._async_client.get(self._make_key(key))
        return self._deserialize(data)

    async def set_async(
        self, key: str, state: dict[str, Any], ttl: int | None = None
    ) -> None:
        """Set the state for a circuit breaker (async)."""
        if self._async_client is None:
            raise RuntimeError("Async client not configured. Use set instead.")
        full_key = self._make_key(key)
        serialized = self._serialize(state)
        effective_ttl = ttl if ttl is not None else self._default_ttl

        if effective_ttl is not None:
            await self._async_client.setex(full_key, effective_ttl, serialized)
        else:
            await self._async_client.set(full_key, serialized)

    async def delete_async(self, key: str) -> None:
        """Delete the state for a circuit breaker (async)."""
        if self._async_client is None:
            raise RuntimeError("Async client not configured. Use delete instead.")
        await self._async_client.delete(self._make_key(key))

    def close(self) -> None:
        """Close sync Redis connection."""
        if self._sync_client is not None:
            self._sync_client.close()

    async def close_async(self) -> None:
        """Close async Redis connection."""
        if self._async_client is not None:
            await self._async_client.close()
