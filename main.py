"""Example usage of c-breaker circuit breaker library."""

import random

from cbreaker import CircuitOpenError, CircuitState, DetectorType, circuit_breaker


# Simple example with time-based detection
@circuit_breaker(
    name="example_service",
    detector_type=DetectorType.TIME_BASED,
    failure_threshold=3,
    time_window_seconds=60,
    recovery_timeout=10.0,
)
def call_external_service() -> dict:
    """Simulate an external service call that might fail."""
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Service unavailable")
    return {"status": "success", "data": "Hello from service!"}


def on_state_change(old_state: CircuitState, new_state: CircuitState) -> None:
    """Callback when circuit state changes."""
    print(f"  [State Change] {old_state.value} -> {new_state.value}")


# Example with sliding window and fallback
@circuit_breaker(
    name="api_with_fallback",
    detector_type=DetectorType.SLIDING_WINDOW,
    window_size=5,
    failure_rate_threshold=0.5,
    min_calls=3,
    on_state_change=on_state_change,
    fallback=lambda: {"status": "cached", "message": "Using cached response"},
)
def call_api_with_fallback() -> dict:
    """Simulate an API call with fallback response."""
    if random.random() < 0.6:
        raise TimeoutError("Request timed out")
    return {"status": "success", "data": "Fresh data from API"}


def main() -> None:
    """Run circuit breaker examples."""
    print("=" * 60)
    print("C-Breaker Circuit Breaker Library - Examples")
    print("=" * 60)

    # Example 1: Basic usage
    print("\n--- Example 1: Time-based Circuit Breaker ---")
    for i in range(10):
        try:
            result = call_external_service()
            print(f"Call {i + 1}: Success - {result}")
        except CircuitOpenError as e:
            print(f"Call {i + 1}: Circuit OPEN - {e}")
        except ConnectionError as e:
            print(f"Call {i + 1}: Failed - {e}")

    # Access circuit breaker stats
    cb = call_external_service.circuit_breaker
    print(f"\nCircuit breaker stats: {cb.get_stats()}")

    # Example 2: With fallback
    print("\n--- Example 2: Sliding Window with Fallback ---")
    for i in range(10):
        try:
            result = call_api_with_fallback()
            print(f"Call {i + 1}: {result}")
        except TimeoutError as e:
            print(f"Call {i + 1}: Failed - {e}")

    # Access circuit breaker
    cb2 = call_api_with_fallback.circuit_breaker
    print(f"\nCircuit breaker stats: {cb2.get_stats()}")


if __name__ == "__main__":
    main()
