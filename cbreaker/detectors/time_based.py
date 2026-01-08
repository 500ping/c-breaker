"""Time-based failure detection strategy."""

import threading
import time
from typing import Any

from cbreaker.detectors.base import BaseFailureDetector


class TimeBasedFailureDetector(BaseFailureDetector):
    """
    Time-based failure detection.

    Counts failures within a fixed time window. The circuit trips if the
    failure count exceeds the threshold within the time window.

    Supports both in-memory storage (with thread locking) and Redis storage
    for distributed applications.

    Example (in-memory):
        detector = TimeBasedFailureDetector(
            failure_threshold=5,
            time_window_seconds=60
        )
        # Circuit trips if 5+ failures occur within 60 seconds

    Example (Redis):
        import redis
        redis_client = redis.Redis(host='localhost', port=6379)
        detector = TimeBasedFailureDetector(
            name="my_service",
            failure_threshold=5,
            time_window_seconds=60,
            sync_client=redis_client
        )
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        time_window_seconds: float = 60.0,
        name: str | None = None,
        sync_client: Any | None = None,
        async_client: Any | None = None,
        key_prefix: str = "cbreaker:detector:",
    ):
        """
        Initialize the time-based failure detector.

        Args:
            failure_threshold: Number of failures to trip the circuit.
            time_window_seconds: Time window in seconds to count failures.
            name: Unique name for this detector (required for Redis storage).
            sync_client: Synchronous Redis client (redis-py).
            async_client: Asynchronous Redis client (redis.asyncio).
            key_prefix: Prefix for Redis keys.
        """
        self.failure_threshold = failure_threshold
        self.time_window_seconds = time_window_seconds

        # Redis configuration
        self._name = name
        self._sync_client = sync_client
        self._async_client = async_client
        self._key_prefix = key_prefix
        self._use_redis = sync_client is not None or async_client is not None

        if self._use_redis and name is None:
            raise ValueError("name is required when using Redis storage")

        # In-memory storage with thread lock
        self._failures: list[float] = []
        self._lock = threading.RLock()

    @property
    def _failures_key(self) -> str:
        """Get the Redis key for storing failures."""
        return f"{self._key_prefix}{self._name}:failures"

    def _check_sync_client(self) -> None:
        """Verify sync client is available when using Redis."""
        if self._use_redis and self._sync_client is None:
            raise RuntimeError("Sync client not configured. Use async methods.")

    def _check_async_client(self) -> None:
        """Verify async client is available when using Redis."""
        if self._use_redis and self._async_client is None:
            raise RuntimeError("Async client not configured. Use sync methods.")

    def _cleanup_old_failures(self, current_time: float | None = None) -> None:
        """Remove failures outside the time window (in-memory)."""
        if current_time is None:
            current_time = time.time()
        cutoff = current_time - self.time_window_seconds
        self._failures = [ts for ts in self._failures if ts > cutoff]

    def _cleanup_old_failures_redis(self, current_time: float | None = None) -> None:
        """Remove failures outside the time window (Redis sync)."""
        if self._sync_client is None:
            return
        if current_time is None:
            current_time = time.time()
        cutoff = current_time - self.time_window_seconds
        self._sync_client.zremrangebyscore(self._failures_key, "-inf", cutoff)

    async def _cleanup_old_failures_redis_async(
        self, current_time: float | None = None
    ) -> None:
        """Remove failures outside the time window (Redis async)."""
        if self._async_client is None:
            return
        if current_time is None:
            current_time = time.time()
        cutoff = current_time - self.time_window_seconds
        await self._async_client.zremrangebyscore(self._failures_key, "-inf", cutoff)

    def _cleanup_failures(self, current_time: float | None = None) -> None:
        """Remove failures outside the time window."""
        if self._use_redis:
            self._cleanup_old_failures_redis(current_time)
        else:
            with self._lock:
                self._cleanup_old_failures(current_time)

    async def _cleanup_failures_async(self, current_time: float | None = None) -> None:
        """Remove failures outside the time window (async)."""
        if self._use_redis:
            await self._cleanup_old_failures_redis_async(current_time)
        else:
            with self._lock:
                self._cleanup_old_failures(current_time)

    def _add_failure(
        self, timestamp: float, exception: Exception | None = None
    ) -> None:
        """Add a failure record."""
        if self._use_redis:
            member = f"{timestamp}:{id(exception) if exception else 0}"
            self._sync_client.zadd(self._failures_key, {member: timestamp})
        else:
            with self._lock:
                self._failures.append(timestamp)

    async def _add_failure_async(
        self, timestamp: float, exception: Exception | None = None
    ) -> None:
        """Add a failure record (async)."""
        if self._use_redis:
            member = f"{timestamp}:{id(exception) if exception else 0}"
            await self._async_client.zadd(self._failures_key, {member: timestamp})
        else:
            with self._lock:
                self._failures.append(timestamp)

    def _get_failure_count(self) -> int:
        """Get current failure count within the time window."""
        if self._use_redis:
            return self._sync_client.zcard(self._failures_key)
        with self._lock:
            return len(self._failures)

    async def _get_failure_count_async(self) -> int:
        """Get current failure count within the time window (async)."""
        if self._use_redis:
            return await self._async_client.zcard(self._failures_key)
        with self._lock:
            return len(self._failures)

    def record_success(self, timestamp: float) -> None:
        """Record a successful call."""
        self._check_sync_client()
        self._cleanup_failures(timestamp)

    async def record_success_async(self, timestamp: float) -> None:
        """Record a successful call (async)."""
        self._check_async_client()
        await self._cleanup_failures_async(timestamp)

    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call."""
        self._check_sync_client()
        self._cleanup_failures(timestamp)
        self._add_failure(timestamp, exception)

    async def record_failure_async(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call (async)."""
        self._check_async_client()
        await self._cleanup_failures_async(timestamp)
        await self._add_failure_async(timestamp, exception)

    def should_trip(self) -> bool:
        """Check if failures exceed threshold within time window."""
        self._check_sync_client()
        self._cleanup_failures()
        return self._get_failure_count() >= self.failure_threshold

    async def should_trip_async(self) -> bool:
        """Check if failures exceed threshold (async)."""
        self._check_async_client()
        await self._cleanup_failures_async()
        return await self._get_failure_count_async() >= self.failure_threshold

    def reset(self) -> None:
        """Reset the detector state."""
        self._check_sync_client()
        if self._use_redis:
            self._sync_client.delete(self._failures_key)
        else:
            with self._lock:
                self._failures.clear()

    async def reset_async(self) -> None:
        """Reset the detector state (async)."""
        self._check_async_client()
        if self._use_redis:
            await self._async_client.delete(self._failures_key)
        else:
            with self._lock:
                self._failures.clear()

    def _get_failures_list(self) -> list[float]:
        """Get list of failure timestamps."""
        if self._use_redis:
            self._cleanup_old_failures_redis()
            members = self._sync_client.zrange(
                self._failures_key, 0, -1, withscores=True
            )
            return [score for _, score in members]
        with self._lock:
            self._cleanup_old_failures()
            return self._failures.copy()

    def get_state(self) -> dict[str, Any]:
        """Get the current state for serialization."""
        return {
            "failures": self._get_failures_list()
            if not self._use_redis or self._sync_client
            else [],
            "failure_threshold": self.failure_threshold,
            "time_window_seconds": self.time_window_seconds,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        failures = state.get("failures", [])
        current_time = time.time()
        cutoff = current_time - self.time_window_seconds

        if self._use_redis and self._sync_client:
            self._sync_client.delete(self._failures_key)
            for i, ts in enumerate(failures):
                if ts > cutoff:
                    self._sync_client.zadd(self._failures_key, {f"{ts}:{i}": ts})
        else:
            with self._lock:
                self._failures = [ts for ts in failures if ts > cutoff]

    @property
    def failure_count(self) -> int:
        """Get the current failure count within the time window."""
        self._check_sync_client()
        self._cleanup_failures()
        return self._get_failure_count()
