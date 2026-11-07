"""Structured logging and performance monitoring utilities."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator

# Format: timestamp - logger name - level - [function:line] - message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "EdgeTrack", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = setup_logger("EdgeTrack")


@contextmanager
def time_block(operation_name: str) -> Generator[dict, None, None]:
    """Context manager for high-precision latency profiling (in milliseconds)."""
    metric = {"operation": operation_name, "duration_ms": 0.0}
    start_time = time.perf_counter()
    try:
        yield metric
    finally:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        metric["duration_ms"] = round(elapsed, 3)

