"""
c-breaker: A flexible circuit breaker library for Python.

This library provides a circuit breaker pattern implementation with:
- Multiple failure detection strategies (time-based, sliding window, combined, custom)
- Redis support for horizontal scaling
- Both sync and async support
- Easy-to-use decorator interface
"""

from cbreaker.core.breaker import CircuitBreaker
from cbreaker.core.states import CircuitState
from cbreaker.decorators.circuit import circuit_breaker
from cbreaker.detectors.base import BaseFailureDetector
from cbreaker.detectors.combined import CombinedFailureDetector
from cbreaker.detectors.sliding_window import SlidingWindowFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector
from cbreaker.enums import DetectorType
from cbreaker.exceptions import CircuitBreakerError, CircuitOpenError
from cbreaker.storage.base import BaseStorage
from cbreaker.storage.memory import MemoryStorage
from cbreaker.storage.redis_storage import RedisStorage

__version__ = "0.1.0"

__all__ = [
    # Core
    "CircuitBreaker",
    "CircuitState",
    # Enums
    "DetectorType",
    # Decorators
    "circuit_breaker",
    # Detectors
    "BaseFailureDetector",
    "TimeBasedFailureDetector",
    "SlidingWindowFailureDetector",
    "CombinedFailureDetector",
    # Storage
    "BaseStorage",
    "MemoryStorage",
    "RedisStorage",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
]
