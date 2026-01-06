"""Combined failure detection strategy."""

from typing import Any

from cbreaker.detectors.base import BaseFailureDetector
from cbreaker.detectors.sliding_window import SlidingWindowFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector


class CombinedFailureDetector(BaseFailureDetector):
    """
    Combined time-based and sliding window failure detection.

    Trips the circuit if EITHER the time-based OR sliding window
    detector indicates a trip condition. This provides comprehensive
    protection against both burst failures and sustained failure rates.

    Example:
        detector = CombinedFailureDetector(
            failure_threshold=5,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5
        )
        # Circuit trips if:
        # - 5+ failures in 60 seconds, OR
        # - 50%+ failure rate in last 10 calls
    """

    def __init__(
        self,
        # Time-based parameters
        failure_threshold: int = 5,
        time_window_seconds: float = 60.0,
        # Sliding window parameters
        window_size: int = 10,
        failure_rate_threshold: float = 0.5,
        min_calls: int = 5,
        # Combination mode
        require_both: bool = False,
    ):
        """
        Initialize the combined failure detector.

        Args:
            failure_threshold: Number of failures to trip (time-based).
            time_window_seconds: Time window in seconds (time-based).
            window_size: Number of calls to track (sliding window).
            failure_rate_threshold: Failure rate threshold (sliding window).
            min_calls: Minimum calls before tripping (sliding window).
            require_both: If True, both detectors must trip. If False, either.
        """
        self._time_based = TimeBasedFailureDetector(
            failure_threshold=failure_threshold,
            time_window_seconds=time_window_seconds,
        )
        self._sliding_window = SlidingWindowFailureDetector(
            window_size=window_size,
            failure_rate_threshold=failure_rate_threshold,
            min_calls=min_calls,
        )
        self.require_both = require_both

    def record_success(self, timestamp: float) -> None:
        """Record a successful call to both detectors."""
        self._time_based.record_success(timestamp)
        self._sliding_window.record_success(timestamp)

    def record_failure(
        self,
        timestamp: float,
        exception: Exception | None = None,
    ) -> None:
        """Record a failed call to both detectors."""
        self._time_based.record_failure(timestamp, exception)
        self._sliding_window.record_failure(timestamp, exception)

    def should_trip(self) -> bool:
        """Check if circuit should trip based on combination mode."""
        time_based_trip = self._time_based.should_trip()
        sliding_window_trip = self._sliding_window.should_trip()

        if self.require_both:
            return time_based_trip and sliding_window_trip
        return time_based_trip or sliding_window_trip

    def reset(self) -> None:
        """Reset both detectors."""
        self._time_based.reset()
        self._sliding_window.reset()

    def get_state(self) -> dict[str, Any]:
        """Get the combined state for serialization."""
        return {
            "time_based": self._time_based.get_state(),
            "sliding_window": self._sliding_window.get_state(),
            "require_both": self.require_both,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        if "time_based" in state:
            self._time_based.load_state(state["time_based"])
        if "sliding_window" in state:
            self._sliding_window.load_state(state["sliding_window"])

    @property
    def time_based_detector(self) -> TimeBasedFailureDetector:
        """Access the underlying time-based detector."""
        return self._time_based

    @property
    def sliding_window_detector(self) -> SlidingWindowFailureDetector:
        """Access the underlying sliding window detector."""
        return self._sliding_window
