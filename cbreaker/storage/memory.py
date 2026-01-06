"""In-memory storage backend."""

import threading
from typing import Any

from cbreaker.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """
    In-memory storage backend.

    Stores circuit breaker state in memory. Suitable for single-instance
    applications or testing. State is lost on application restart.

    Thread-safe implementation using locks.

    Example:
        storage = MemoryStorage()
        breaker = CircuitBreaker(name="my_service", storage=storage)
    """

    def __init__(self) -> None:
        """Initialize the memory storage."""
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        """Get the state for a circuit breaker."""
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, state: dict[str, Any], ttl: int | None = None) -> None:
        """
        Set the state for a circuit breaker.

        Note: TTL is ignored in memory storage (no automatic expiry).
        """
        with self._lock:
            self._data[key] = state.copy()

    def delete(self, key: str) -> None:
        """Delete the state for a circuit breaker."""
        with self._lock:
            self._data.pop(key, None)

    async def get_async(self, key: str) -> dict[str, Any] | None:
        """Async version of get."""
        return self.get(key)

    async def set_async(
        self, key: str, state: dict[str, Any], ttl: int | None = None
    ) -> None:
        """Async version of set."""
        self.set(key, state, ttl)

    async def delete_async(self, key: str) -> None:
        """Async version of delete."""
        self.delete(key)

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._data.clear()

    def keys(self) -> list[str]:
        """Get all stored keys."""
        with self._lock:
            return list(self._data.keys())
