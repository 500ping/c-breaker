"""Base class for failure detection strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FailureEvent:
    """Represents a failure or success event."""

    timestamp: float
    is_failure: bool
    exception: Exception | None = None
    metadata: dict[str, Any] | None = None


class BaseFailureDetector(ABC):
    """
    Abstract base class for failure detection strategies.

    Implement this class to create custom failure detection logic.
    """

    @abstractmethod
    def record_success(self, timestamp: float) -> None:
        """
        Record a successful call.

        Args:
            timestamp: The Unix timestamp when the success occurred.
        """
        pass

    @abstractmethod
    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """
        Record a failed call.

        Args:
            timestamp: The Unix timestamp when the failure occurred.
            exception: The exception that caused the failure, if any.
        """
        pass

    @abstractmethod
    def should_trip(self) -> bool:
        """
        Determine if the circuit should trip (open).

        Returns:
            True if the failure threshold has been exceeded, False otherwise.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector state (called when circuit closes)."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Get the current state of the detector for serialization.

        Returns:
            A dictionary representing the current state.
        """
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load state from a dictionary (for deserialization).

        Args:
            state: The state dictionary to load.
        """
        pass

    async def record_success_async(self, timestamp: float) -> None:
        """Async version of record_success. Default delegates to sync version."""
        self.record_success(timestamp)

    async def record_failure_async(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Async version of record_failure. Default delegates to sync version."""
        self.record_failure(timestamp, exception)

    async def should_trip_async(self) -> bool:
        """Async version of should_trip. Default delegates to sync version."""
        return self.should_trip()

    async def reset_async(self) -> None:
        """Async version of reset. Default delegates to sync version."""
        self.reset()
