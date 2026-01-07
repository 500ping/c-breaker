"""Sliding window failure detection strategy."""

import json
import threading
from collections import deque
from typing import Any

from cbreaker.detectors.base import BaseFailureDetector


class SlidingWindowFailureDetector(BaseFailureDetector):
    """
    Sliding window failure detection.

    Tracks the last N calls and trips if the failure rate exceeds the threshold.
    This provides a more balanced view compared to time-based detection.

    Supports both in-memory storage (with thread locking) and Redis storage
    for distributed applications.

    Example (in-memory):
        detector = SlidingWindowFailureDetector(
            window_size=10,
            failure_rate_threshold=0.5
        )
        # Circuit trips if 50%+ of the last 10 calls failed

    Example (Redis):
        import redis
        redis_client = redis.Redis(host='localhost', port=6379)
        detector = SlidingWindowFailureDetector(
            name="my_service",
            window_size=10,
            failure_rate_threshold=0.5,
            sync_client=redis_client
        )
    """

    def __init__(
        self,
        window_size: int = 10,
        failure_rate_threshold: float = 0.5,
        min_calls: int = 5,
        name: str | None = None,
        sync_client: Any | None = None,
        async_client: Any | None = None,
        key_prefix: str = "cbreaker:detector:",
    ):
        """
        Initialize the sliding window failure detector.

        Args:
            window_size: Number of calls to track in the window.
            failure_rate_threshold: Failure rate (0.0-1.0) to trip the circuit.
            min_calls: Minimum number of calls before tripping is possible.
            name: Unique name for this detector (required for Redis storage).
            sync_client: Synchronous Redis client (redis-py).
            async_client: Asynchronous Redis client (redis.asyncio).
            key_prefix: Prefix for Redis keys.
        """
        if not 0.0 <= failure_rate_threshold <= 1.0:
            raise ValueError("failure_rate_threshold must be between 0.0 and 1.0")
        if min_calls > window_size:
            raise ValueError("min_calls cannot be greater than window_size")

        self.window_size = window_size
        self.failure_rate_threshold = failure_rate_threshold
        self.min_calls = min_calls

        # Redis configuration
        self._name = name
        self._sync_client = sync_client
        self._async_client = async_client
        self._key_prefix = key_prefix
        self._use_redis = sync_client is not None or async_client is not None

        if self._use_redis and name is None:
            raise ValueError("name is required when using Redis storage")

        # In-memory storage with thread lock
        # deque with (timestamp, is_failure) tuples
        self._calls: deque[tuple[float, bool]] = deque(maxlen=window_size)
        self._lock = threading.RLock()

    @property
    def _calls_key(self) -> str:
        """Get the Redis key for storing calls."""
        return f"{self._key_prefix}{self._name}:calls"

    def _record_call_redis(self, timestamp: float, is_failure: bool) -> None:
        """Record a call to Redis list (sync)."""
        if self._sync_client is None:
            raise RuntimeError("Sync client not configured.")
        record = json.dumps({"ts": timestamp, "f": is_failure})
        pipe = self._sync_client.pipeline()
        pipe.rpush(self._calls_key, record)
        pipe.ltrim(self._calls_key, -self.window_size, -1)
        pipe.execute()

    async def _record_call_redis_async(
        self, timestamp: float, is_failure: bool
    ) -> None:
        """Record a call to Redis list (async)."""
        if self._async_client is None:
            raise RuntimeError("Async client not configured.")
        record = json.dumps({"ts": timestamp, "f": is_failure})
        pipe = self._async_client.pipeline()
        pipe.rpush(self._calls_key, record)
        pipe.ltrim(self._calls_key, -self.window_size, -1)
        await pipe.execute()

    def _get_calls_redis(self) -> list[tuple[float, bool]]:
        """Get all calls from Redis (sync)."""
        if self._sync_client is None:
            raise RuntimeError("Sync client not configured.")
        records = self._sync_client.lrange(self._calls_key, 0, -1)
        calls = []
        for record in records:
            if isinstance(record, bytes):
                record = record.decode("utf-8")
            data = json.loads(record)
            calls.append((data["ts"], data["f"]))
        return calls

    async def _get_calls_redis_async(self) -> list[tuple[float, bool]]:
        """Get all calls from Redis (async)."""
        if self._async_client is None:
            raise RuntimeError("Async client not configured.")
        records = await self._async_client.lrange(self._calls_key, 0, -1)
        calls = []
        for record in records:
            if isinstance(record, bytes):
                record = record.decode("utf-8")
            data = json.loads(record)
            calls.append((data["ts"], data["f"]))
        return calls

    def record_success(self, timestamp: float) -> None:
        """Record a successful call."""
        if self._use_redis:
            if self._sync_client is None:
                raise RuntimeError(
                    "Sync client not configured. Use record_success_async."
                )
            self._record_call_redis(timestamp, False)
        else:
            with self._lock:
                self._calls.append((timestamp, False))

    async def record_success_async(self, timestamp: float) -> None:
        """Record a successful call (async)."""
        if self._use_redis:
            if self._async_client is None:
                raise RuntimeError("Async client not configured. Use record_success.")
            await self._record_call_redis_async(timestamp, False)
        else:
            with self._lock:
                self._calls.append((timestamp, False))

    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call."""
        if self._use_redis:
            if self._sync_client is None:
                raise RuntimeError(
                    "Sync client not configured. Use record_failure_async."
                )
            self._record_call_redis(timestamp, True)
        else:
            with self._lock:
                self._calls.append((timestamp, True))

    async def record_failure_async(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call (async)."""
        if self._use_redis:
            if self._async_client is None:
                raise RuntimeError("Async client not configured. Use record_failure.")
            await self._record_call_redis_async(timestamp, True)
        else:
            with self._lock:
                self._calls.append((timestamp, True))

    def should_trip(self) -> bool:
        """Check if failure rate exceeds threshold."""
        if self._use_redis:
            if self._sync_client is None:
                raise RuntimeError("Sync client not configured. Use should_trip_async.")
            calls = self._get_calls_redis()
        else:
            with self._lock:
                calls = list(self._calls)

        if len(calls) < self.min_calls:
            return False

        failure_count = sum(1 for _, is_failure in calls if is_failure)
        failure_rate = failure_count / len(calls)
        return failure_rate >= self.failure_rate_threshold

    async def should_trip_async(self) -> bool:
        """Check if failure rate exceeds threshold (async)."""
        if self._use_redis:
            if self._async_client is None:
                raise RuntimeError("Async client not configured. Use should_trip.")
            calls = await self._get_calls_redis_async()
        else:
            with self._lock:
                calls = list(self._calls)

        if len(calls) < self.min_calls:
            return False

        failure_count = sum(1 for _, is_failure in calls if is_failure)
        failure_rate = failure_count / len(calls)
        return failure_rate >= self.failure_rate_threshold

    def reset(self) -> None:
        """Reset the detector state."""
        if self._use_redis:
            if self._sync_client is None:
                raise RuntimeError("Sync client not configured. Use reset_async.")
            self._sync_client.delete(self._calls_key)
        else:
            with self._lock:
                self._calls.clear()

    async def reset_async(self) -> None:
        """Reset the detector state (async)."""
        if self._use_redis:
            if self._async_client is None:
                raise RuntimeError("Async client not configured. Use reset.")
            await self._async_client.delete(self._calls_key)
        else:
            with self._lock:
                self._calls.clear()

    def get_state(self) -> dict[str, Any]:
        """Get the current state for serialization."""
        if self._use_redis and self._sync_client is not None:
            calls = self._get_calls_redis()
        else:
            with self._lock:
                calls = list(self._calls)

        return {
            "calls": calls,
            "window_size": self.window_size,
            "failure_rate_threshold": self.failure_rate_threshold,
            "min_calls": self.min_calls,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        calls = state.get("calls", [])

        if self._use_redis and self._sync_client is not None:
            self._sync_client.delete(self._calls_key)
            pipe = self._sync_client.pipeline()
            for ts, is_failure in calls[-self.window_size :]:
                record = json.dumps({"ts": ts, "f": is_failure})
                pipe.rpush(self._calls_key, record)
            pipe.execute()
        else:
            with self._lock:
                self._calls.clear()
                for call in calls[-self.window_size :]:
                    self._calls.append(tuple(call))

    @property
    def failure_rate(self) -> float:
        """Get the current failure rate."""
        if self._use_redis and self._sync_client is not None:
            calls = self._get_calls_redis()
        else:
            with self._lock:
                calls = list(self._calls)

        if not calls:
            return 0.0
        failure_count = sum(1 for _, is_failure in calls if is_failure)
        return failure_count / len(calls)

    @property
    def call_count(self) -> int:
        """Get the current number of calls in the window."""
        if self._use_redis:
            if self._sync_client is None:
                raise RuntimeError("Sync client not configured.")
            return self._sync_client.llen(self._calls_key)
        else:
            with self._lock:
                return len(self._calls)
