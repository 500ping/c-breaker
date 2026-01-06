"""Enums for the circuit breaker library."""

from enum import Enum


class DetectorType(str, Enum):
    """Enumeration of failure detector types."""

    TIME_BASED = "time_based"
    SLIDING_WINDOW = "sliding_window"
    COMBINED = "combined"
