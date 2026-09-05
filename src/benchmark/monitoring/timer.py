"""Precision latency and execution timer."""

import time
from typing import Dict, Any


class BenchmarkTimer:
    """Precision timer for tracking benchmark execution latency and throughput."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed_sec(self) -> float:
        if self.end_time >= self.start_time and self.start_time > 0:
            return round(self.end_time - self.start_time, 4)
        return round(time.perf_counter() - self.start_time, 4)
