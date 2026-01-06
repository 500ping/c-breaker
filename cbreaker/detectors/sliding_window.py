"""Sliding window failure detection strategy."""

from collections import deque
from typing import Any

from cbreaker.detectors.base import BaseFailureDetector


class SlidingWindowFailureDetector(BaseFailureDetector):
    """
    Sliding window failure detection.

    Tracks the last N calls and trips if the failure rate exceeds the threshold.
    This provides a more balanced view compared to time-based detection.

    Example:
        detector = SlidingWindowFailureDetector(
            window_size=10,
            failure_rate_threshold=0.5
        )
        # Circuit trips if 50%+ of the last 10 calls failed
    """

    def __init__(
        self,
        window_size: int = 10,
        failure_rate_threshold: float = 0.5,
        min_calls: int = 5,
    ):
        """
        Initialize the sliding window failure detector.

        Args:
            window_size: Number of calls to track in the window.
            failure_rate_threshold: Failure rate (0.0-1.0) to trip the circuit.
            min_calls: Minimum number of calls before tripping is possible.
        """
        if not 0.0 <= failure_rate_threshold <= 1.0:
            raise ValueError("failure_rate_threshold must be between 0.0 and 1.0")
        if min_calls > window_size:
            raise ValueError("min_calls cannot be greater than window_size")

        self.window_size = window_size
        self.failure_rate_threshold = failure_rate_threshold
        self.min_calls = min_calls
        # deque with (timestamp, is_failure) tuples
        self._calls: deque[tuple[float, bool]] = deque(maxlen=window_size)

    def record_success(self, timestamp: float) -> None:
        """Record a successful call."""
        self._calls.append((timestamp, False))

    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call."""
        self._calls.append((timestamp, True))

    def should_trip(self) -> bool:
        """Check if failure rate exceeds threshold."""
        if len(self._calls) < self.min_calls:
            return False

        failure_count = sum(1 for _, is_failure in self._calls if is_failure)
        failure_rate = failure_count / len(self._calls)
        return failure_rate >= self.failure_rate_threshold

    def reset(self) -> None:
        """Reset the detector state."""
        self._calls.clear()

    def get_state(self) -> dict[str, Any]:
        """Get the current state for serialization."""
        return {
            "calls": list(self._calls),
            "window_size": self.window_size,
            "failure_rate_threshold": self.failure_rate_threshold,
            "min_calls": self.min_calls,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        calls = state.get("calls", [])
        self._calls.clear()
        for call in calls[-self.window_size :]:
            self._calls.append(tuple(call))

    @property
    def failure_rate(self) -> float:
        """Get the current failure rate."""
        if not self._calls:
            return 0.0
        failure_count = sum(1 for _, is_failure in self._calls if is_failure)
        return failure_count / len(self._calls)

    @property
    def call_count(self) -> int:
        """Get the current number of calls in the window."""
        return len(self._calls)
