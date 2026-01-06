"""Failure detection strategies."""

from cbreaker.detectors.base import BaseFailureDetector
from cbreaker.detectors.combined import CombinedFailureDetector
from cbreaker.detectors.sliding_window import SlidingWindowFailureDetector
from cbreaker.detectors.time_based import TimeBasedFailureDetector

__all__ = [
    "BaseFailureDetector",
    "TimeBasedFailureDetector",
    "SlidingWindowFailureDetector",
    "CombinedFailureDetector",
]
