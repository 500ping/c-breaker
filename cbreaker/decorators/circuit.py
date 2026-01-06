"""Circuit breaker decorator implementation."""

import asyncio
import functools
from collections.abc import Callable
from typing import Any, TypeVar

from cbreaker.core.breaker import CircuitBreaker
from cbreaker.core.states import CircuitState
from cbreaker.detectors.base import BaseFailureDetector
from cbreaker.detectors.combined import CombinedFailureDetector
from cbreaker.detectors.sliding_window import SlidingWindowFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector
from cbreaker.enums import DetectorType
from cbreaker.exceptions import CircuitOpenError
from cbreaker.storage.base import BaseStorage
from cbreaker.storage.memory import MemoryStorage

F = TypeVar("F", bound=Callable[..., Any])

# Global registry of circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker | None:
    """
    Get a circuit breaker by name from the global registry.

    Args:
        name: The name of the circuit breaker.

    Returns:
        The circuit breaker instance, or None if not found.
    """
    return _circuit_breakers.get(name)


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """
    Get all registered circuit breakers.

    Returns:
        Dictionary of all circuit breakers by name.
    """
    return _circuit_breakers.copy()


def circuit_breaker(
    name: str | None = None,
    # Failure detection strategy
    detector: BaseFailureDetector | None = None,
    detector_type: DetectorType | str = DetectorType.TIME_BASED,
    # Time-based detector options
    failure_threshold: int = 5,
    time_window_seconds: float = 60.0,
    # Sliding window detector options
    window_size: int = 10,
    failure_rate_threshold: float = 0.5,
    min_calls: int = 5,
    # Combined detector options
    require_both: bool = False,
    # Circuit breaker options
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    excluded_exceptions: tuple[type[Exception], ...] | None = None,
    # Storage
    storage: BaseStorage | None = None,
    # Callbacks
    on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    fallback: Callable[..., Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a circuit breaker.

    Automatically detects if the function is sync or async and handles accordingly.

    Args:
        name: Unique name for this circuit breaker. Defaults to function name.
        detector: Custom failure detector instance. Overrides detector_type.
        detector_type: Type of failure detector to use (DetectorType enum or string):
            - DetectorType.TIME_BASED: Count failures within a time window
            - DetectorType.SLIDING_WINDOW: Track failure rate in last N calls
            - DetectorType.COMBINED: Both time-based and sliding window
        failure_threshold: Failures to trip circuit (time-based/combined).
        time_window_seconds: Time window in seconds (time-based/combined).
        window_size: Number of calls to track (sliding_window/combined).
        failure_rate_threshold: Failure rate to trip (sliding_window/combined).
        min_calls: Min calls before tripping (sliding_window/combined).
        require_both: For combined, require both detectors to trip.
        recovery_timeout: Seconds before OPEN -> HALF_OPEN transition.
        half_open_max_calls: Max calls in HALF_OPEN state.
        excluded_exceptions: Exceptions that don't count as failures.
        storage: Storage backend for state persistence.
        on_state_change: Callback when state changes.
        fallback: Function to call when circuit is open.

    Example (simple):
        @circuit_breaker(name="my_api")
        def call_api():
            return requests.get("https://api.example.com")

    Example (async with options):
        @circuit_breaker(
            name="my_api",
            detector_type="sliding_window",
            window_size=20,
            failure_rate_threshold=0.3,
            fallback=lambda: {"error": "Service unavailable"}
        )
        async def call_api():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.example.com") as response:
                    return await response.json()

    Example (custom detector):
        class MyDetector(BaseFailureDetector):
            # Custom implementation
            pass

        @circuit_breaker(name="custom", detector=MyDetector())
        def my_function():
            pass
    """

    def decorator(func: F) -> F:
        # Determine circuit breaker name
        cb_name = name or func.__qualname__

        # Normalize detector_type to enum
        effective_detector_type = (
            DetectorType(detector_type)
            if isinstance(detector_type, str)
            else detector_type
        )

        # Create failure detector
        if detector is not None:
            failure_detector = detector
        elif effective_detector_type == DetectorType.SLIDING_WINDOW:
            failure_detector = SlidingWindowFailureDetector(
                window_size=window_size,
                failure_rate_threshold=failure_rate_threshold,
                min_calls=min_calls,
            )
        elif effective_detector_type == DetectorType.COMBINED:
            failure_detector = CombinedFailureDetector(
                failure_threshold=failure_threshold,
                time_window_seconds=time_window_seconds,
                window_size=window_size,
                failure_rate_threshold=failure_rate_threshold,
                min_calls=min_calls,
                require_both=require_both,
            )
        else:  # default: TIME_BASED
            failure_detector = TimeBasedFailureDetector(
                failure_threshold=failure_threshold,
                time_window_seconds=time_window_seconds,
            )

        # Create storage
        cb_storage = storage or MemoryStorage()

        # Create circuit breaker
        cb = CircuitBreaker(
            name=cb_name,
            failure_detector=failure_detector,
            storage=cb_storage,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            excluded_exceptions=excluded_exceptions,
            on_state_change=on_state_change,
        )

        # Register in global registry
        _circuit_breakers[cb_name] = cb

        if asyncio.iscoroutinefunction(func):
            # Async wrapper
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await cb.call_async(func, *args, **kwargs)
                except CircuitOpenError:
                    if fallback is not None:
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        return fallback(*args, **kwargs)
                    raise

            # Attach circuit breaker to wrapper for inspection
            async_wrapper.circuit_breaker = cb  # type: ignore
            return async_wrapper  # type: ignore
        else:
            # Sync wrapper
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return cb.call(func, *args, **kwargs)
                except CircuitOpenError:
                    if fallback is not None:
                        return fallback(*args, **kwargs)
                    raise

            # Attach circuit breaker to wrapper for inspection
            sync_wrapper.circuit_breaker = cb  # type: ignore
            return sync_wrapper  # type: ignore

    return decorator
