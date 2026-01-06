"""Circuit breaker states."""

from enum import Enum


class CircuitState(str, Enum):
    """Enumeration of circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is tripped, requests are rejected
    HALF_OPEN = "half_open"  # Testing if the service has recovered
