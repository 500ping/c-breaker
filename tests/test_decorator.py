"""Tests for the circuit breaker decorator."""

import pytest

from cbreaker.core.states import CircuitState
from cbreaker.decorators.circuit import (
    circuit_breaker,
    get_all_circuit_breakers,
    get_circuit_breaker,
)
from cbreaker.enums import DetectorType


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""

    def test_basic_decorator(self) -> None:
        """Decorator should wrap function with circuit breaker."""

        @circuit_breaker(name="test_basic")
        def my_func():
            return "result"

        result = my_func()
        assert result == "result"

    def test_decorator_preserves_function_name(self) -> None:
        """Decorator should preserve function metadata."""

        @circuit_breaker(name="test_meta")
        def my_named_func():
            """My docstring."""
            pass

        assert my_named_func.__name__ == "my_named_func"
        assert my_named_func.__doc__ == "My docstring."

    def test_circuit_breaker_attached(self) -> None:
        """Circuit breaker should be accessible from decorated function."""

        @circuit_breaker(name="test_attached")
        def my_func():
            pass

        assert hasattr(my_func, "circuit_breaker")
        assert my_func.circuit_breaker.name == "test_attached"

    def test_default_name_from_function(self) -> None:
        """Default name should be derived from function name."""

        @circuit_breaker()
        def unique_function_name():
            pass

        cb = unique_function_name.circuit_breaker
        assert "unique_function_name" in cb.name

    def test_fallback_on_open(self) -> None:
        """Fallback should be called when circuit is open."""

        @circuit_breaker(
            name="test_fallback",
            failure_threshold=1,
            fallback=lambda: "fallback_result",
        )
        def failing_func():
            raise RuntimeError("fail")

        # First call fails and trips circuit
        with pytest.raises(RuntimeError):
            failing_func()

        # Second call uses fallback
        result = failing_func()
        assert result == "fallback_result"

    def test_detector_type_time_based(self) -> None:
        """Time-based detector should be used."""

        @circuit_breaker(
            name="test_time_based",
            detector_type="time_based",
            failure_threshold=3,
        )
        def my_func():
            pass

        cb = my_func.circuit_breaker
        assert "TimeBasedFailureDetector" in type(cb._failure_detector).__name__

    def test_detector_type_sliding_window(self) -> None:
        """Sliding window detector should be used."""

        @circuit_breaker(
            name="test_sliding",
            detector_type=DetectorType.SLIDING_WINDOW,
            window_size=10,
            failure_rate_threshold=0.5,
        )
        def my_func():
            pass

        cb = my_func.circuit_breaker
        assert "SlidingWindowFailureDetector" in type(cb._failure_detector).__name__

    def test_detector_type_combined(self) -> None:
        """Combined detector should be used."""

        @circuit_breaker(
            name="test_combined",
            detector_type=DetectorType.COMBINED,
        )
        def my_func():
            pass

        cb = my_func.circuit_breaker
        assert "CombinedFailureDetector" in type(cb._failure_detector).__name__

    def test_global_registry(self) -> None:
        """Circuit breakers should be registered globally."""

        @circuit_breaker(name="registered_cb")
        def my_func():
            pass

        cb = get_circuit_breaker("registered_cb")
        assert cb is not None
        assert cb.name == "registered_cb"

        all_cbs = get_all_circuit_breakers()
        assert "registered_cb" in all_cbs

    def test_state_change_callback(self) -> None:
        """State change callback should be called."""
        changes: list[tuple[CircuitState, CircuitState]] = []

        def on_change(old: CircuitState, new: CircuitState) -> None:
            changes.append((old, new))

        @circuit_breaker(
            name="test_callback",
            failure_threshold=1,
            on_state_change=on_change,
        )
        def failing_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            failing_func()

        assert len(changes) == 1
        assert changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)


class TestAsyncDecorator:
    """Tests for async function decoration."""

    @pytest.mark.asyncio
    async def test_async_decorator(self) -> None:
        """Decorator should work with async functions."""

        @circuit_breaker(name="test_async")
        async def async_func():
            return "async_result"

        result = await async_func()
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_fallback(self) -> None:
        """Async fallback should work."""

        async def fallback():
            return "async_fallback"

        @circuit_breaker(
            name="test_async_fallback",
            failure_threshold=1,
            fallback=fallback,
        )
        async def failing_async():
            raise RuntimeError("fail")

        # First call fails
        with pytest.raises(RuntimeError):
            await failing_async()

        # Second call uses fallback
        result = await failing_async()
        assert result == "async_fallback"

    @pytest.mark.asyncio
    async def test_async_trips_circuit(self) -> None:
        """Async failures should trip circuit."""

        @circuit_breaker(
            name="test_async_trip",
            failure_threshold=2,
        )
        async def failing_async():
            raise OSError("fail")

        for _ in range(2):
            with pytest.raises(IOError):
                await failing_async()

        cb = failing_async.circuit_breaker
        assert cb.is_open
