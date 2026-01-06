"""Tests for failure detectors."""

import time

import pytest

from cbreaker.detectors.combined import CombinedFailureDetector
from cbreaker.detectors.sliding_window import SlidingWindowFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector


class TestTimeBasedFailureDetector:
    """Tests for TimeBasedFailureDetector."""

    def test_should_not_trip_below_threshold(self) -> None:
        """Circuit should not trip when failures are below threshold."""
        detector = TimeBasedFailureDetector(failure_threshold=5, time_window_seconds=60)
        now = time.time()

        for i in range(4):
            detector.record_failure(now + i)

        assert not detector.should_trip()
        assert detector.failure_count == 4

    def test_should_trip_at_threshold(self) -> None:
        """Circuit should trip when failures reach threshold."""
        detector = TimeBasedFailureDetector(failure_threshold=5, time_window_seconds=60)
        now = time.time()

        for i in range(5):
            detector.record_failure(now + i)

        assert detector.should_trip()
        assert detector.failure_count == 5

    def test_old_failures_are_cleaned_up(self) -> None:
        """Failures outside time window should be removed."""
        detector = TimeBasedFailureDetector(failure_threshold=5, time_window_seconds=10)
        now = time.time()

        # Record old failures
        for i in range(3):
            detector.record_failure(now - 20 + i)

        # Record recent failures
        for i in range(2):
            detector.record_failure(now + i)

        assert detector.failure_count == 2
        assert not detector.should_trip()

    def test_reset_clears_failures(self) -> None:
        """Reset should clear all recorded failures."""
        detector = TimeBasedFailureDetector(failure_threshold=5, time_window_seconds=60)
        now = time.time()

        for i in range(5):
            detector.record_failure(now + i)

        assert detector.should_trip()
        detector.reset()
        assert not detector.should_trip()
        assert detector.failure_count == 0

    def test_state_serialization(self) -> None:
        """State should be serializable and restorable."""
        detector = TimeBasedFailureDetector(failure_threshold=5, time_window_seconds=60)
        now = time.time()

        for i in range(3):
            detector.record_failure(now + i)

        state = detector.get_state()
        assert "failures" in state

        new_detector = TimeBasedFailureDetector(
            failure_threshold=5, time_window_seconds=60
        )
        new_detector.load_state(state)
        assert new_detector.failure_count == 3


class TestSlidingWindowFailureDetector:
    """Tests for SlidingWindowFailureDetector."""

    def test_should_not_trip_below_threshold(self) -> None:
        """Circuit should not trip when failure rate is below threshold."""
        detector = SlidingWindowFailureDetector(
            window_size=10, failure_rate_threshold=0.5, min_calls=5
        )
        now = time.time()

        # 4 successes, 1 failure = 20% failure rate
        for i in range(4):
            detector.record_success(now + i)
        detector.record_failure(now + 4)

        assert not detector.should_trip()
        assert detector.failure_rate == 0.2

    def test_should_trip_at_threshold(self) -> None:
        """Circuit should trip when failure rate reaches threshold."""
        detector = SlidingWindowFailureDetector(
            window_size=10, failure_rate_threshold=0.5, min_calls=5
        )
        now = time.time()

        # 3 successes, 3 failures = 50% failure rate
        for i in range(3):
            detector.record_success(now + i)
        for i in range(3):
            detector.record_failure(now + 3 + i)

        assert detector.should_trip()
        assert detector.failure_rate == 0.5

    def test_min_calls_required(self) -> None:
        """Circuit should not trip before min_calls is reached."""
        detector = SlidingWindowFailureDetector(
            window_size=10, failure_rate_threshold=0.5, min_calls=5
        )
        now = time.time()

        # 4 failures, but min_calls is 5
        for i in range(4):
            detector.record_failure(now + i)

        assert not detector.should_trip()
        assert detector.failure_rate == 1.0

    def test_window_slides(self) -> None:
        """Oldest calls should be removed when window is full."""
        detector = SlidingWindowFailureDetector(
            window_size=5, failure_rate_threshold=0.5, min_calls=3
        )
        now = time.time()

        # Fill window with failures
        for i in range(5):
            detector.record_failure(now + i)

        assert detector.should_trip()

        # Add successes to push out failures
        for i in range(5):
            detector.record_success(now + 5 + i)

        assert not detector.should_trip()
        assert detector.failure_rate == 0.0

    def test_invalid_parameters(self) -> None:
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowFailureDetector(failure_rate_threshold=1.5)

        with pytest.raises(ValueError):
            SlidingWindowFailureDetector(window_size=5, min_calls=10)


class TestCombinedFailureDetector:
    """Tests for CombinedFailureDetector."""

    def test_trips_on_time_based(self) -> None:
        """Should trip when time-based condition is met."""
        detector = CombinedFailureDetector(
            failure_threshold=3,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.8,
            min_calls=5,
            require_both=False,
        )
        now = time.time()

        # 3 failures should trip time-based but not sliding window
        for i in range(3):
            detector.record_failure(now + i)

        assert detector.should_trip()

    def test_trips_on_sliding_window(self) -> None:
        """Should trip when sliding window condition is met."""
        detector = CombinedFailureDetector(
            failure_threshold=10,  # High threshold
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=4,
            require_both=False,
        )
        now = time.time()

        # 2 successes, 3 failures = 60% failure rate
        for i in range(2):
            detector.record_success(now + i)
        for i in range(3):
            detector.record_failure(now + 2 + i)

        assert detector.should_trip()

    def test_require_both_mode(self) -> None:
        """In require_both mode, both conditions must be met."""
        detector = CombinedFailureDetector(
            failure_threshold=3,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            require_both=True,
        )
        now = time.time()

        # Only time-based is met
        for i in range(3):
            detector.record_failure(now + i)

        # Time-based is met, but not sliding window (only 3 calls, min is 5)
        assert not detector.should_trip()

        # Add more failures to meet sliding window
        for i in range(3):
            detector.record_failure(now + 3 + i)

        # Now both are met
        assert detector.should_trip()
