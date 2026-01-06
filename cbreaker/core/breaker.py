"""Core circuit breaker implementation."""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from cbreaker.core.states import CircuitState
from cbreaker.detectors.base import BaseFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector
from cbreaker.exceptions import CircuitOpenError
from cbreaker.storage.base import BaseStorage
from cbreaker.storage.memory import MemoryStorage


class CircuitBreaker:
    """
    Circuit breaker implementation with pluggable failure detection and storage.

    The circuit breaker pattern prevents cascading failures by temporarily
    stopping calls to a failing service.

    States:
        - CLOSED: Normal operation, calls pass through
        - OPEN: Circuit tripped, calls are rejected
        - HALF_OPEN: Testing recovery, limited calls allowed

    Example:
        breaker = CircuitBreaker(
            name="external_api",
            failure_detector=TimeBasedFailureDetector(
                failure_threshold=5,
                time_window_seconds=60
            ),
            recovery_timeout=30.0
        )

        try:
            result = breaker.call(external_api_function, arg1, arg2)
        except CircuitOpenError:
            # Handle circuit open
            pass
    """

    def __init__(
        self,
        name: str,
        failure_detector: BaseFailureDetector | None = None,
        storage: BaseStorage | None = None,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: tuple[type[Exception], ...] | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Unique name for this circuit breaker.
            failure_detector: Failure detection strategy.
                Defaults to TimeBasedFailureDetector.
            storage: Storage backend for state persistence.
                Defaults to MemoryStorage.
            recovery_timeout: Seconds to wait before transitioning
                from OPEN to HALF_OPEN.
            half_open_max_calls: Max calls allowed in HALF_OPEN state before decision.
            excluded_exceptions: Exception types that should NOT count as failures.
            on_state_change: Callback when state changes (old_state, new_state).
        """
        self.name = name
        self._failure_detector = failure_detector or TimeBasedFailureDetector()
        self._storage = storage or MemoryStorage()
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or ()
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

        # Load state from storage if available
        self._load_state()

    def _load_state(self) -> None:
        """Load state from storage."""
        stored = self._storage.get(self.name)
        if stored:
            self._state = CircuitState(stored.get("state", CircuitState.CLOSED.value))
            self._opened_at = stored.get("opened_at")
            self._half_open_calls = stored.get("half_open_calls", 0)
            if "detector_state" in stored:
                self._failure_detector.load_state(stored["detector_state"])

    async def _load_state_async(self) -> None:
        """Load state from storage (async)."""
        stored = await self._storage.get_async(self.name)
        if stored:
            self._state = CircuitState(stored.get("state", CircuitState.CLOSED.value))
            self._opened_at = stored.get("opened_at")
            self._half_open_calls = stored.get("half_open_calls", 0)
            if "detector_state" in stored:
                self._failure_detector.load_state(stored["detector_state"])

    def _save_state(self) -> None:
        """Save current state to storage."""
        state = {
            "state": self._state.value,
            "opened_at": self._opened_at,
            "half_open_calls": self._half_open_calls,
            "detector_state": self._failure_detector.get_state(),
        }
        self._storage.set(self.name, state)

    async def _save_state_async(self) -> None:
        """Save current state to storage (async)."""
        state = {
            "state": self._state.value,
            "opened_at": self._opened_at,
            "half_open_calls": self._half_open_calls,
            "detector_state": self._failure_detector.get_state(),
        }
        await self._storage.set_async(self.name, state)

    def _set_state(self, new_state: CircuitState) -> None:
        """Set new state and trigger callback if changed."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state

            if new_state == CircuitState.OPEN:
                self._opened_at = time.time()
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self._opened_at = None
                self._half_open_calls = 0
                self._failure_detector.reset()
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0

            self._save_state()

            if self._on_state_change:
                self._on_state_change(old_state, new_state)

    async def _set_state_async(self, new_state: CircuitState) -> None:
        """Set new state and trigger callback if changed (async)."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state

            if new_state == CircuitState.OPEN:
                self._opened_at = time.time()
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self._opened_at = None
                self._half_open_calls = 0
                await self._failure_detector.reset_async()
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0

            await self._save_state_async()

            if self._on_state_change:
                self._on_state_change(old_state, new_state)

    def _get_remaining_timeout(self) -> float | None:
        """Get remaining time until circuit can transition to HALF_OPEN."""
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return None
        elapsed = time.time() - self._opened_at
        remaining = self.recovery_timeout - elapsed
        return max(0, remaining)

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return False
        return time.time() - self._opened_at >= self.recovery_timeout

    def _is_excluded_exception(self, exc: Exception) -> bool:
        """Check if exception should be excluded from failure counting."""
        return isinstance(exc, self.excluded_exceptions)

    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if the request can proceed, False otherwise.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._set_state(CircuitState.HALF_OPEN)
                    return True
                return False

            # HALF_OPEN: allow limited calls
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    async def allow_request_async(self) -> bool:
        """Async version of allow_request."""
        # Use the sync version since state checks are fast
        return self.allow_request()

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            timestamp = time.time()
            self._failure_detector.record_success(timestamp)

            if self._state == CircuitState.HALF_OPEN:
                # Success in HALF_OPEN means we can close the circuit
                self._set_state(CircuitState.CLOSED)
            else:
                self._save_state()

    async def record_success_async(self) -> None:
        """Async version of record_success."""
        timestamp = time.time()
        await self._failure_detector.record_success_async(timestamp)

        if self._state == CircuitState.HALF_OPEN:
            await self._set_state_async(CircuitState.CLOSED)
        else:
            await self._save_state_async()

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record a failed call."""
        with self._lock:
            if exception and self._is_excluded_exception(exception):
                return

            timestamp = time.time()
            self._failure_detector.record_failure(timestamp, exception)

            if self._state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN means we need to open again
                self._set_state(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_detector.should_trip():
                    self._set_state(CircuitState.OPEN)
                else:
                    self._save_state()

    async def record_failure_async(self, exception: Exception | None = None) -> None:
        """Async version of record_failure."""
        if exception and self._is_excluded_exception(exception):
            return

        timestamp = time.time()
        await self._failure_detector.record_failure_async(timestamp, exception)

        if self._state == CircuitState.HALF_OPEN:
            await self._set_state_async(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if await self._failure_detector.should_trip_async():
                await self._set_state_async(CircuitState.OPEN)
            else:
                await self._save_state_async()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        if not self.allow_request():
            raise CircuitOpenError(self.name, self._get_remaining_timeout())

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    async def call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: The async function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        if not await self.allow_request_async():
            raise CircuitOpenError(self.name, self._get_remaining_timeout())

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self.record_success_async()
            return result
        except Exception as e:
            await self.record_failure_async(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._set_state(CircuitState.CLOSED)

    async def reset_async(self) -> None:
        """Async version of reset."""
        await self._set_state_async(CircuitState.CLOSED)

    def trip(self) -> None:
        """Manually trip the circuit breaker to OPEN state."""
        with self._lock:
            self._set_state(CircuitState.OPEN)

    async def trip_async(self) -> None:
        """Async version of trip."""
        await self._set_state_async(CircuitState.OPEN)

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "opened_at": self._opened_at,
                "remaining_timeout": self._get_remaining_timeout(),
                "half_open_calls": self._half_open_calls,
                "detector_state": self._failure_detector.get_state(),
            }

    def __repr__(self) -> str:
        """String representation of the circuit breaker."""
        return f"CircuitBreaker(name={self.name!r}, state={self._state.value})"
