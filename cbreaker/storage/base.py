"""Base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Any


class BaseStorage(ABC):
    """
    Abstract base class for circuit breaker state storage.

    Implement this class to create custom storage backends.
    Storage is used to persist circuit breaker state across restarts
    and share state across multiple instances (e.g., using Redis).
    """

    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None:
        """
        Get the state for a circuit breaker.

        Args:
            key: The unique identifier for the circuit breaker.

        Returns:
            The state dictionary, or None if not found.
        """
        pass

    @abstractmethod
    def set(self, key: str, state: dict[str, Any], ttl: int | None = None) -> None:
        """
        Set the state for a circuit breaker.

        Args:
            key: The unique identifier for the circuit breaker.
            state: The state dictionary to store.
            ttl: Optional time-to-live in seconds.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete the state for a circuit breaker.

        Args:
            key: The unique identifier for the circuit breaker.
        """
        pass

    @abstractmethod
    async def get_async(self, key: str) -> dict[str, Any] | None:
        """
        Async version of get.

        Args:
            key: The unique identifier for the circuit breaker.

        Returns:
            The state dictionary, or None if not found.
        """
        pass

    @abstractmethod
    async def set_async(
        self, key: str, state: dict[str, Any], ttl: int | None = None
    ) -> None:
        """
        Async version of set.

        Args:
            key: The unique identifier for the circuit breaker.
            state: The state dictionary to store.
            ttl: Optional time-to-live in seconds.
        """
        pass

    @abstractmethod
    async def delete_async(self, key: str) -> None:
        """
        Async version of delete.

        Args:
            key: The unique identifier for the circuit breaker.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any connections. Override if needed."""
        pass

    @abstractmethod
    async def close_async(self) -> None:
        """Async version of close. Override if needed."""
        pass
