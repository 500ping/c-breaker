"""Tests for the core circuit breaker."""

import pytest
from fakeredis import FakeRedis, FakeServer
from fakeredis.aioredis import FakeRedis as FakeAsyncRedis

from cbreaker.core.breaker import CircuitBreaker
from cbreaker.core.states import CircuitState
from cbreaker.detectors.time_based import TimeBasedFailureDetector
from cbreaker.exceptions import CircuitOpenError
from cbreaker.storage.memory import MemoryStorage
from cbreaker.storage.redis_storage import RedisStorage


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self) -> None:
        """Circuit should start in CLOSED state."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    def test_successful_call(self) -> None:
        """Successful calls should pass through."""
        cb = CircuitBreaker(name="test")

        result = cb.call(lambda: "success")

        assert result == "success"
        assert cb.is_closed

    def test_failed_call_raises_exception(self) -> None:
        """Failed calls should re-raise the exception."""
        cb = CircuitBreaker(name="test")

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            cb.call(failing_func)

    def test_circuit_trips_after_threshold(self) -> None:
        """Circuit should open after failure threshold is reached."""
        cb = CircuitBreaker(
            name="test",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=3, time_window_seconds=60
            ),
        )

        def failing_func():
            raise RuntimeError("fail")

        # First 3 failures
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        assert cb.is_open

    def test_open_circuit_rejects_calls(self) -> None:
        """Open circuit should reject calls with CircuitOpenError."""
        cb = CircuitBreaker(
            name="test",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=1, time_window_seconds=60
            ),
        )

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.is_open

        # Next call should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: "success")

        assert exc_info.value.circuit_name == "test"

    def test_excluded_exceptions_not_counted(self) -> None:
        """Excluded exceptions should not count as failures."""
        cb = CircuitBreaker(
            name="test",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=2, time_window_seconds=60
            ),
            excluded_exceptions=(ValueError,),
        )

        def raise_value_error():
            raise ValueError("excluded")

        # These shouldn't trip the circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                cb.call(raise_value_error)

        assert cb.is_closed

    def test_manual_trip(self) -> None:
        """Manual trip should open the circuit."""
        cb = CircuitBreaker(name="test")

        cb.trip()

        assert cb.is_open

    def test_manual_reset(self) -> None:
        """Manual reset should close the circuit."""
        cb = CircuitBreaker(name="test")
        cb.trip()

        cb.reset()

        assert cb.is_closed

    def test_state_change_callback(self) -> None:
        """State change callback should be called."""
        state_changes: list[tuple[CircuitState, CircuitState]] = []

        def on_change(old: CircuitState, new: CircuitState) -> None:
            state_changes.append((old, new))

        cb = CircuitBreaker(
            name="test",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=1, time_window_seconds=60
            ),
            on_state_change=on_change,
        )

        # Trip the circuit
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_get_stats(self) -> None:
        """get_stats should return circuit info."""
        cb = CircuitBreaker(name="test_stats")
        stats = cb.get_stats()

        assert stats["name"] == "test_stats"
        assert stats["state"] == "closed"
        assert "detector_state" in stats

    def test_repr(self) -> None:
        """repr should return meaningful string."""
        cb = CircuitBreaker(name="my_circuit")
        assert "my_circuit" in repr(cb)
        assert "closed" in repr(cb)

    def test_state_persistence(self) -> None:
        """State should be persisted to storage."""
        storage = MemoryStorage()
        cb = CircuitBreaker(name="persistent", storage=storage)

        cb.trip()

        # Create new breaker with same storage
        cb2 = CircuitBreaker(name="persistent", storage=storage)

        assert cb2.is_open


class TestCircuitBreakerAsync:
    """Async tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_async_successful_call(self) -> None:
        """Async successful calls should pass through."""
        cb = CircuitBreaker(name="async_test")

        async def async_func():
            return "async_success"

        result = await cb.call_async(async_func)

        assert result == "async_success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_async_failed_call(self) -> None:
        """Async failed calls should trip circuit."""
        cb = CircuitBreaker(
            name="async_test",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=2, time_window_seconds=60
            ),
        )

        async def failing_async():
            raise OSError("async fail")

        for _ in range(2):
            with pytest.raises(IOError):
                await cb.call_async(failing_async)

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_async_open_rejects(self) -> None:
        """Open circuit should reject async calls."""
        cb = CircuitBreaker(name="async_test")
        cb.trip()

        async def async_func():
            return "success"

        with pytest.raises(CircuitOpenError):
            await cb.call_async(async_func)

    @pytest.mark.asyncio
    async def test_async_reset(self) -> None:
        """Async reset should work."""
        cb = CircuitBreaker(name="async_test")
        cb.trip()

        await cb.reset_async()

        assert cb.is_closed


class TestCircuitBreakerRedis:
    """Tests for CircuitBreaker with Redis storage."""

    @pytest.fixture
    def fake_server(self) -> FakeServer:
        """Create a fake Redis server for sharing state."""
        return FakeServer()

    @pytest.fixture
    def redis_client(self, fake_server: FakeServer) -> FakeRedis:
        """Create a fake Redis client."""
        return FakeRedis(server=fake_server, decode_responses=True)

    @pytest.fixture
    def async_redis_client(self, fake_server: FakeServer) -> FakeAsyncRedis:
        """Create a fake async Redis client sharing same server."""
        return FakeAsyncRedis(server=fake_server, decode_responses=True)

    @pytest.fixture
    def redis_storage(
        self, redis_client: FakeRedis, async_redis_client: FakeAsyncRedis
    ) -> RedisStorage:
        """Create a Redis storage instance."""
        return RedisStorage(
            sync_client=redis_client,
            async_client=async_redis_client,
        )

    def test_redis_storage_detected(self, redis_storage: RedisStorage) -> None:
        """CircuitBreaker should detect Redis storage."""
        cb = CircuitBreaker(name="redis_test", storage=redis_storage)
        assert cb._use_redis is True

    def test_memory_storage_not_redis(self) -> None:
        """CircuitBreaker should not set _use_redis for memory storage."""
        cb = CircuitBreaker(name="memory_test", storage=MemoryStorage())
        assert cb._use_redis is False

    def test_state_shared_between_instances(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Two circuit breaker instances should share state via Redis."""
        storage1 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )

        detector1 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="shared_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )
        detector2 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="shared_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )

        cb1 = CircuitBreaker(
            name="shared_test", storage=storage1, failure_detector=detector1
        )
        cb2 = CircuitBreaker(
            name="shared_test", storage=storage2, failure_detector=detector2
        )

        # Initially both should be closed
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

        # Trip cb1
        cb1.trip()

        # cb2 should see the state change
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

    def test_state_isolation_between_different_names(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Different circuit breaker names should have isolated state."""
        storage = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )

        cb1 = CircuitBreaker(name="circuit_a", storage=storage)
        cb2 = CircuitBreaker(name="circuit_b", storage=storage)

        # Trip cb1
        cb1.trip()

        # cb2 should still be closed
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.CLOSED

    def test_reset_clears_redis_state(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Reset should clear state in Redis."""
        storage = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        cb1 = CircuitBreaker(name="reset_test", storage=storage)

        # Create second instance to verify shared state
        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        cb2 = CircuitBreaker(name="reset_test", storage=storage2)

        # Trip and reset from cb1
        cb1.trip()
        assert cb2.state == CircuitState.OPEN

        cb1.reset()

        # cb2 should see the reset
        assert cb2.state == CircuitState.CLOSED

    def test_get_stats_from_redis(self, redis_storage: RedisStorage) -> None:
        """get_stats should return current state from Redis."""
        cb = CircuitBreaker(name="stats_test", storage=redis_storage)

        cb.trip()
        stats = cb.get_stats()

        assert stats["state"] == "open"
        assert stats["opened_at"] is not None
        assert stats["remaining_timeout"] is not None

    def test_repr_uses_redis_state(self, redis_storage: RedisStorage) -> None:
        """__repr__ should use Redis state."""
        cb = CircuitBreaker(name="repr_test", storage=redis_storage)

        cb.trip()
        repr_str = repr(cb)

        assert "open" in repr_str
        assert "repr_test" in repr_str

    def test_half_open_calls_shared(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Half-open call count should be shared via Redis."""
        storage = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        detector = TimeBasedFailureDetector(
            failure_threshold=1,
            time_window_seconds=60,
            name="half_open_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )

        cb1 = CircuitBreaker(
            name="half_open_test",
            storage=storage,
            failure_detector=detector,
            recovery_timeout=0,  # Immediate recovery
            half_open_max_calls=2,
        )

        # Trip the circuit
        cb1.trip()
        assert cb1.state == CircuitState.OPEN

        # First request should transition to half-open and be allowed
        assert cb1.allow_request() is True
        assert cb1.state == CircuitState.HALF_OPEN

        # Create second instance to check shared half_open_calls
        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        detector2 = TimeBasedFailureDetector(
            failure_threshold=1,
            time_window_seconds=60,
            name="half_open_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )
        cb2 = CircuitBreaker(
            name="half_open_test",
            storage=storage2,
            failure_detector=detector2,
            recovery_timeout=0,
            half_open_max_calls=2,
        )

        # Second instance should see half-open state
        assert cb2.state == CircuitState.HALF_OPEN

        # Second instance should be allowed (2nd of 2 max calls)
        assert cb2.allow_request() is True

        # Third request should be rejected (max calls reached)
        assert cb1.allow_request() is False
        assert cb2.allow_request() is False


class TestCircuitBreakerRedisAsync:
    """Async tests for CircuitBreaker with Redis storage."""

    @pytest.fixture
    def fake_server(self) -> FakeServer:
        """Create a fake Redis server for sharing state."""
        return FakeServer()

    @pytest.fixture
    def redis_client(self, fake_server: FakeServer) -> FakeRedis:
        """Create a fake Redis client."""
        return FakeRedis(server=fake_server, decode_responses=True)

    @pytest.fixture
    def async_redis_client(self, fake_server: FakeServer) -> FakeAsyncRedis:
        """Create a fake async Redis client sharing same server."""
        return FakeAsyncRedis(server=fake_server, decode_responses=True)

    @pytest.mark.asyncio
    async def test_async_state_shared(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Async operations should share state via Redis."""
        storage1 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )

        detector1 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="async_shared_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )
        detector2 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="async_shared_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )

        cb1 = CircuitBreaker(
            name="async_shared_test", storage=storage1, failure_detector=detector1
        )
        cb2 = CircuitBreaker(
            name="async_shared_test", storage=storage2, failure_detector=detector2
        )

        await cb1.trip_async()

        # cb2 should see the state change
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_reset_shared(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Async reset should be visible to other instances."""
        storage1 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )

        cb1 = CircuitBreaker(name="async_reset_test", storage=storage1)
        cb2 = CircuitBreaker(name="async_reset_test", storage=storage2)

        cb1.trip()
        assert cb2.state == CircuitState.OPEN

        await cb2.reset_async()
        assert cb1.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_record_failure_shared(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Async failure recording should affect shared state."""
        storage = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        detector1 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="async_failure_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )
        detector2 = TimeBasedFailureDetector(
            failure_threshold=2,
            time_window_seconds=60,
            name="async_failure_detector",
            sync_client=redis_client,
            async_client=async_redis_client,
        )

        storage2 = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )

        cb1 = CircuitBreaker(
            name="async_failure_test", storage=storage, failure_detector=detector1
        )
        cb2 = CircuitBreaker(
            name="async_failure_test", storage=storage2, failure_detector=detector2
        )

        # Record failure from cb1
        await cb1.record_failure_async(RuntimeError("fail 1"))
        assert cb1.state == CircuitState.CLOSED

        # Record failure from cb2 (should trip the circuit)
        await cb2.record_failure_async(RuntimeError("fail 2"))

        # Both should see OPEN state
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_allow_request(
        self,
        redis_client: FakeRedis,
        async_redis_client: FakeAsyncRedis,
    ) -> None:
        """Async allow_request should work with Redis."""
        storage = RedisStorage(
            sync_client=redis_client, async_client=async_redis_client
        )
        cb = CircuitBreaker(
            name="async_allow_test",
            storage=storage,
            recovery_timeout=0,
        )

        # Closed circuit allows requests
        assert await cb.allow_request_async() is True

        # Trip the circuit
        await cb.trip_async()

        # With recovery_timeout=0, it should transition to half-open
        assert await cb.allow_request_async() is True
        assert cb.state == CircuitState.HALF_OPEN
