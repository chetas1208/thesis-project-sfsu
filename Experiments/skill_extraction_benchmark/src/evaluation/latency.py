"""Latency measurement utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict


@contextmanager
def timer():
    """Context manager that yields a dict which will contain 'elapsed_seconds'."""
    result: Dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 4)
