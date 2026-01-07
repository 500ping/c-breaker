"""Tests for failure detectors."""

import time

import fakeredis
import fakeredis.aioredis
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


class TestTimeBasedFailureDetectorRedis:
    """Tests for TimeBasedFailureDetector with Redis storage."""

    def test_requires_name_for_redis(self) -> None:
        """Should raise ValueError if name is not provided with Redis client."""

        redis_client = fakeredis.FakeRedis()
        with pytest.raises(ValueError, match="name is required"):
            TimeBasedFailureDetector(
                failure_threshold=5,
                time_window_seconds=60,
                sync_client=redis_client,
            )

    def test_should_trip_at_threshold_redis(self) -> None:
        """Circuit should trip when failures reach threshold (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = TimeBasedFailureDetector(
            name="test_time_based",
            failure_threshold=5,
            time_window_seconds=60,
            sync_client=redis_client,
        )
        now = time.time()

        for i in range(5):
            detector.record_failure(now + i)

        assert detector.should_trip()
        assert detector.failure_count == 5

    def test_old_failures_cleaned_up_redis(self) -> None:
        """Failures outside time window should be removed (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = TimeBasedFailureDetector(
            name="test_cleanup",
            failure_threshold=5,
            time_window_seconds=10,
            sync_client=redis_client,
        )
        now = time.time()

        # Record old failures
        for i in range(3):
            detector.record_failure(now - 20 + i)

        # Record recent failures
        for i in range(2):
            detector.record_failure(now + i)

        assert detector.failure_count == 2
        assert not detector.should_trip()

    def test_reset_clears_redis(self) -> None:
        """Reset should clear all recorded failures in Redis."""

        redis_client = fakeredis.FakeRedis()
        detector = TimeBasedFailureDetector(
            name="test_reset",
            failure_threshold=5,
            time_window_seconds=60,
            sync_client=redis_client,
        )
        now = time.time()

        for i in range(5):
            detector.record_failure(now + i)

        assert detector.should_trip()
        detector.reset()
        assert not detector.should_trip()
        assert detector.failure_count == 0

    def test_shared_state_between_instances(self) -> None:
        """Multiple detector instances should share state via Redis."""

        redis_client = fakeredis.FakeRedis()
        detector1 = TimeBasedFailureDetector(
            name="shared_detector",
            failure_threshold=5,
            time_window_seconds=60,
            sync_client=redis_client,
        )
        detector2 = TimeBasedFailureDetector(
            name="shared_detector",
            failure_threshold=5,
            time_window_seconds=60,
            sync_client=redis_client,
        )
        now = time.time()

        # Record failures via detector1
        for i in range(3):
            detector1.record_failure(now + i)

        # Record failures via detector2
        for i in range(2):
            detector2.record_failure(now + 3 + i)

        # Both should see all 5 failures
        assert detector1.failure_count == 5
        assert detector2.failure_count == 5
        assert detector1.should_trip()
        assert detector2.should_trip()


class TestSlidingWindowFailureDetectorRedis:
    """Tests for SlidingWindowFailureDetector with Redis storage."""

    def test_requires_name_for_redis(self) -> None:
        """Should raise ValueError if name is not provided with Redis client."""

        redis_client = fakeredis.FakeRedis()
        with pytest.raises(ValueError, match="name is required"):
            SlidingWindowFailureDetector(
                window_size=10,
                failure_rate_threshold=0.5,
                sync_client=redis_client,
            )

    def test_should_trip_at_threshold_redis(self) -> None:
        """Circuit should trip when failure rate reaches threshold (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = SlidingWindowFailureDetector(
            name="test_sliding",
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        now = time.time()

        # 3 successes, 3 failures = 50% failure rate
        for i in range(3):
            detector.record_success(now + i)
        for i in range(3):
            detector.record_failure(now + 3 + i)

        assert detector.should_trip()
        assert detector.failure_rate == 0.5

    def test_window_slides_redis(self) -> None:
        """Oldest calls should be removed when window is full (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = SlidingWindowFailureDetector(
            name="test_slide",
            window_size=5,
            failure_rate_threshold=0.5,
            min_calls=3,
            sync_client=redis_client,
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

    def test_reset_clears_redis(self) -> None:
        """Reset should clear all recorded calls in Redis."""

        redis_client = fakeredis.FakeRedis()
        detector = SlidingWindowFailureDetector(
            name="test_reset_sliding",
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        now = time.time()

        for i in range(6):
            detector.record_failure(now + i)

        assert detector.should_trip()
        detector.reset()
        assert not detector.should_trip()
        assert detector.call_count == 0

    def test_shared_state_between_instances(self) -> None:
        """Multiple detector instances should share state via Redis."""

        redis_client = fakeredis.FakeRedis()
        detector1 = SlidingWindowFailureDetector(
            name="shared_sliding",
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        detector2 = SlidingWindowFailureDetector(
            name="shared_sliding",
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        now = time.time()

        # Record via detector1
        for i in range(3):
            detector1.record_success(now + i)

        # Record via detector2
        for i in range(3):
            detector2.record_failure(now + 3 + i)

        # Both should see all 6 calls with 50% failure rate
        assert detector1.call_count == 6
        assert detector2.call_count == 6
        assert detector1.failure_rate == 0.5
        assert detector2.failure_rate == 0.5


class TestCombinedFailureDetectorRedis:
    """Tests for CombinedFailureDetector with Redis storage."""

    def test_requires_name_for_redis(self) -> None:
        """Should raise ValueError if name is not provided with Redis client."""

        redis_client = fakeredis.FakeRedis()
        with pytest.raises(ValueError, match="name is required"):
            CombinedFailureDetector(
                failure_threshold=5,
                time_window_seconds=60,
                sync_client=redis_client,
            )

    def test_trips_on_time_based_redis(self) -> None:
        """Should trip when time-based condition is met (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = CombinedFailureDetector(
            name="test_combined",
            failure_threshold=3,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.8,
            min_calls=5,
            require_both=False,
            sync_client=redis_client,
        )
        now = time.time()

        # 3 failures should trip time-based but not sliding window
        for i in range(3):
            detector.record_failure(now + i)

        assert detector.should_trip()

    def test_require_both_mode_redis(self) -> None:
        """In require_both mode, both conditions must be met (Redis)."""

        redis_client = fakeredis.FakeRedis()
        detector = CombinedFailureDetector(
            name="test_both",
            failure_threshold=3,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            require_both=True,
            sync_client=redis_client,
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

    def test_shared_state_between_instances(self) -> None:
        """Multiple detector instances should share state via Redis."""

        redis_client = fakeredis.FakeRedis()
        detector1 = CombinedFailureDetector(
            name="shared_combined",
            failure_threshold=5,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        detector2 = CombinedFailureDetector(
            name="shared_combined",
            failure_threshold=5,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            sync_client=redis_client,
        )
        now = time.time()

        # Record failures via detector1
        for i in range(3):
            detector1.record_failure(now + i)

        # Record failures via detector2
        for i in range(2):
            detector2.record_failure(now + 3 + i)

        # Both should see all 5 failures and trip
        assert detector1.should_trip()
        assert detector2.should_trip()


class TestDetectorsAsync:
    """Tests for async detector methods."""

    @pytest.mark.asyncio
    async def test_time_based_async_redis(self) -> None:
        """Test async methods with Redis storage."""

        redis_client = fakeredis.aioredis.FakeRedis()
        detector = TimeBasedFailureDetector(
            name="test_async",
            failure_threshold=5,
            time_window_seconds=60,
            async_client=redis_client,
        )
        now = time.time()

        for i in range(5):
            await detector.record_failure_async(now + i)

        assert await detector.should_trip_async()

        await detector.reset_async()
        assert not await detector.should_trip_async()

    @pytest.mark.asyncio
    async def test_sliding_window_async_redis(self) -> None:
        """Test async methods with Redis storage."""

        redis_client = fakeredis.aioredis.FakeRedis()
        detector = SlidingWindowFailureDetector(
            name="test_async_sliding",
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            async_client=redis_client,
        )
        now = time.time()

        for i in range(3):
            await detector.record_success_async(now + i)
        for i in range(3):
            await detector.record_failure_async(now + 3 + i)

        assert await detector.should_trip_async()

        await detector.reset_async()
        assert not await detector.should_trip_async()

    @pytest.mark.asyncio
    async def test_combined_async_redis(self) -> None:
        """Test async methods with Redis storage."""

        redis_client = fakeredis.aioredis.FakeRedis()
        detector = CombinedFailureDetector(
            name="test_async_combined",
            failure_threshold=3,
            time_window_seconds=60,
            window_size=10,
            failure_rate_threshold=0.5,
            min_calls=5,
            async_client=redis_client,
        )
        now = time.time()

        for i in range(3):
            await detector.record_failure_async(now + i)

        assert await detector.should_trip_async()

        await detector.reset_async()
        assert not await detector.should_trip_async()

    @pytest.mark.asyncio
    async def test_memory_async_still_works(self) -> None:
        """Test async methods still work with in-memory storage."""
        detector = TimeBasedFailureDetector(
            failure_threshold=5,
            time_window_seconds=60,
        )
        now = time.time()

        for i in range(5):
            await detector.record_failure_async(now + i)

        assert await detector.should_trip_async()

        await detector.reset_async()
        assert not await detector.should_trip_async()
