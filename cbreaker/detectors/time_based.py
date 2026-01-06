"""Time-based failure detection strategy."""

import time
from typing import Any

from cbreaker.detectors.base import BaseFailureDetector


class TimeBasedFailureDetector(BaseFailureDetector):
    """
    Time-based failure detection.

    Counts failures within a fixed time window. The circuit trips if the
    failure count exceeds the threshold within the time window.

    Example:
        detector = TimeBasedFailureDetector(
            failure_threshold=5,
            time_window_seconds=60
        )
        # Circuit trips if 5+ failures occur within 60 seconds
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        time_window_seconds: float = 60.0,
    ):
        """
        Initialize the time-based failure detector.

        Args:
            failure_threshold: Number of failures to trip the circuit.
            time_window_seconds: Time window in seconds to count failures.
        """
        self.failure_threshold = failure_threshold
        self.time_window_seconds = time_window_seconds
        self._failures: list[float] = []  # List of failure timestamps

    def _cleanup_old_failures(self, current_time: float | None = None) -> None:
        """Remove failures outside the time window."""
        if current_time is None:
            current_time = time.time()
        cutoff = current_time - self.time_window_seconds
        self._failures = [ts for ts in self._failures if ts > cutoff]

    def record_success(self, timestamp: float) -> None:
        """Record a successful call."""
        self._cleanup_old_failures(timestamp)

    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call."""
        self._cleanup_old_failures(timestamp)
        self._failures.append(timestamp)

    def should_trip(self) -> bool:
        """Check if failures exceed threshold within time window."""
        self._cleanup_old_failures()
        return len(self._failures) >= self.failure_threshold

    def reset(self) -> None:
        """Reset the detector state."""
        self._failures.clear()

    def get_state(self) -> dict[str, Any]:
        """Get the current state for serialization."""
        return {
            "failures": self._failures.copy(),
            "failure_threshold": self.failure_threshold,
            "time_window_seconds": self.time_window_seconds,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        self._failures = state.get("failures", [])
        self._cleanup_old_failures()

    @property
    def failure_count(self) -> int:
        """Get the current failure count within the time window."""
        self._cleanup_old_failures()
        return len(self._failures)
