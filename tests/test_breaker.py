"""Tests for the core circuit breaker."""

import pytest

from cbreaker.core.breaker import CircuitBreaker
from cbreaker.core.states import CircuitState
from cbreaker.detectors.time_based import TimeBasedFailureDetector
from cbreaker.exceptions import CircuitOpenError
from cbreaker.storage.memory import MemoryStorage


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
