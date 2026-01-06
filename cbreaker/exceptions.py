"""Custom exceptions for the circuit breaker library."""


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when the circuit is open and calls are rejected."""

    def __init__(self, circuit_name: str, remaining_time: float | None = None):
        self.circuit_name = circuit_name
        self.remaining_time = remaining_time
        message = f"Circuit '{circuit_name}' is open"
        if remaining_time is not None:
            message += f", retry after {remaining_time:.2f} seconds"
        super().__init__(message)
