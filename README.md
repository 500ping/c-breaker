# c-breaker

A flexible circuit breaker library for Python with multiple failure detection strategies and Redis support for distributed applications.

## Features

- **4 Failure Detection Strategies**:
  - **Time-based**: Count failures within a time window
  - **Sliding Window**: Track failure rate in last N calls
  - **Combined**: Both time-based and sliding window
  - **Custom**: Implement your own detection logic

- **Distributed Support**: Redis storage backend for horizontal scaling (sync + async)

- **Easy to Use**: Simple decorator interface

- **Async Support**: Full async/await support for modern Python applications

## Installation

```bash
pip install c-breaker

# With Redis support
pip install c-breaker[redis]
```

## Quick Start

### Basic Usage with Decorator

```python
from cbreaker import circuit_breaker

@circuit_breaker(name="my_api")
def call_external_api():
    # Your code here
    return requests.get("https://api.example.com/data")
```

### Async Support

```python
import aiohttp
from cbreaker import circuit_breaker

@circuit_breaker(name="async_api", failure_threshold=3)
async def call_async_api():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            return await response.json()
```

## Failure Detection Strategies

### Time-Based Detection

Trips the circuit if N failures occur within a time window:

```python
from cbreaker import circuit_breaker, DetectorType

@circuit_breaker(
    name="time_based_example",
    detector_type=DetectorType.TIME_BASED,
    failure_threshold=5,      # 5 failures
    time_window_seconds=60,   # within 60 seconds
    recovery_timeout=30.0,    # wait 30s before recovery attempt
)
def my_function():
    pass
```

### Sliding Window Detection

Trips based on failure rate in the last N calls:

```python
from cbreaker import circuit_breaker, DetectorType

@circuit_breaker(
    name="sliding_window_example",
    detector_type=DetectorType.SLIDING_WINDOW,
    window_size=10,               # track last 10 calls
    failure_rate_threshold=0.5,   # trip at 50% failure rate
    min_calls=5,                  # require at least 5 calls
)
def my_function():
    pass
```

### Combined Detection

Uses both strategies - trips if EITHER condition is met:

```python
from cbreaker import circuit_breaker, DetectorType

@circuit_breaker(
    name="combined_example",
    detector_type=DetectorType.COMBINED,
    # Time-based settings
    failure_threshold=5,
    time_window_seconds=60,
    # Sliding window settings
    window_size=10,
    failure_rate_threshold=0.5,
    min_calls=5,
    # Set to True to require BOTH conditions
    require_both=False,
)
def my_function():
    pass
```

### Custom Failure Detector

Create your own detection logic:

```python
from cbreaker import BaseFailureDetector, circuit_breaker
from typing import Any

class MyCustomDetector(BaseFailureDetector):
    def __init__(self, max_failures: int = 3):
        self.max_failures = max_failures
        self.failure_count = 0

    def record_success(self, timestamp: float) -> None:
        self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, timestamp: float, exception: Exception | None = None) -> None:
        self.failure_count += 1

    def should_trip(self) -> bool:
        return self.failure_count >= self.max_failures

    def reset(self) -> None:
        self.failure_count = 0

    def get_state(self) -> dict[str, Any]:
        return {"failure_count": self.failure_count}

    def load_state(self, state: dict[str, Any]) -> None:
        self.failure_count = state.get("failure_count", 0)

@circuit_breaker(name="custom", detector=MyCustomDetector(max_failures=3))
def my_function():
    pass
```

## Redis Storage for Distributed Applications

Share circuit breaker state across multiple application instances:

### Synchronous Redis

```python
import redis
from cbreaker import circuit_breaker, RedisStorage

# Create Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Create storage
storage = RedisStorage(sync_client=redis_client)

@circuit_breaker(
    name="distributed_service",
    storage=storage,
    failure_threshold=5,
)
def call_service():
    pass
```

### Asynchronous Redis

```python
import redis.asyncio as aioredis
from cbreaker import circuit_breaker, RedisStorage

# Create async Redis client
redis_client = aioredis.Redis(host='localhost', port=6379, db=0)

# Create storage with async client
storage = RedisStorage(async_client=redis_client)

@circuit_breaker(
    name="async_distributed_service",
    storage=storage,
)
async def call_service():
    pass
```

## Advanced Usage

### Fallback Function

Execute a fallback when the circuit is open:

```python
def fallback_response(*args, **kwargs):
    return {"status": "service_unavailable", "cached": True}

@circuit_breaker(
    name="with_fallback",
    fallback=fallback_response,
)
def call_api():
    return requests.get("https://api.example.com").json()
```

### Excluded Exceptions

Don't count certain exceptions as failures:

```python
@circuit_breaker(
    name="with_exclusions",
    excluded_exceptions=(ValueError, KeyError),
)
def my_function(value):
    if not value:
        raise ValueError("Empty value")  # Won't count as failure
    return external_call()
```

### State Change Callbacks

React to circuit state changes:

```python
from cbreaker import CircuitState

def on_state_change(old_state: CircuitState, new_state: CircuitState):
    print(f"Circuit changed from {old_state} to {new_state}")
    if new_state == CircuitState.OPEN:
        send_alert("Circuit breaker tripped!")

@circuit_breaker(
    name="with_callback",
    on_state_change=on_state_change,
)
def my_function():
    pass
```

### Manual Circuit Breaker Control

```python
from cbreaker import CircuitBreaker, TimeBasedFailureDetector

# Create circuit breaker manually
breaker = CircuitBreaker(
    name="manual_breaker",
    failure_detector=TimeBasedFailureDetector(
        failure_threshold=5,
        time_window_seconds=60
    ),
    recovery_timeout=30.0,
)

# Use with call method
result = breaker.call(my_function, arg1, arg2)

# Check state
print(f"State: {breaker.state}")
print(f"Is open: {breaker.is_open}")

# Get stats
stats = breaker.get_stats()

# Manual control
breaker.trip()   # Force open
breaker.reset()  # Force close
```

### Access Circuit Breaker from Decorated Function

```python
@circuit_breaker(name="my_service")
def my_function():
    pass

# Access the circuit breaker instance
cb = my_function.circuit_breaker
print(f"Current state: {cb.state}")
print(f"Stats: {cb.get_stats()}")
```

### Global Registry

Access all circuit breakers:

```python
from cbreaker.decorators.circuit import get_circuit_breaker, get_all_circuit_breakers

# Get a specific circuit breaker
cb = get_circuit_breaker("my_service")

# Get all circuit breakers
all_breakers = get_all_circuit_breakers()
for name, breaker in all_breakers.items():
    print(f"{name}: {breaker.state}")
```

## Circuit Breaker States

| State | Description |
|-------|-------------|
| **CLOSED** | Normal operation. Requests pass through. Failures are recorded. |
| **OPEN** | Circuit tripped. Requests are rejected with `CircuitOpenError`. |
| **HALF_OPEN** | Testing recovery. Limited requests allowed to test if service recovered. |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `name` | Function name | Unique identifier for the circuit breaker |
| `detector_type` | `DetectorType.TIME_BASED` | Detection strategy: `DetectorType.TIME_BASED`, `DetectorType.SLIDING_WINDOW`, `DetectorType.COMBINED` (strings also accepted) |
| `failure_threshold` | `5` | Failures to trip (time-based) |
| `time_window_seconds` | `60.0` | Time window in seconds (time-based) |
| `window_size` | `10` | Calls to track (sliding window) |
| `failure_rate_threshold` | `0.5` | Failure rate to trip (sliding window) |
| `min_calls` | `5` | Min calls before tripping (sliding window) |
| `recovery_timeout` | `30.0` | Seconds before OPEN → HALF_OPEN |
| `half_open_max_calls` | `1` | Calls allowed in HALF_OPEN |
| `excluded_exceptions` | `None` | Exceptions that don't count as failures |
| `storage` | `MemoryStorage` | State storage backend |
| `fallback` | `None` | Function to call when circuit is open |
| `on_state_change` | `None` | Callback for state changes |

## License

MIT License
