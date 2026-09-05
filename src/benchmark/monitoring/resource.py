"""Resource monitor tracking process memory, peak RSS, swap, and system memory pressure."""

import os
import psutil
from typing import Dict, Any
from benchmark.hardware import HardwareMonitor


class ResourceMonitor:
    """Monitors process RAM, peak memory, and macOS swap usage during benchmark execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory_gb = self.get_process_memory_gb()
        self.peak_memory_gb = self.baseline_memory_gb

    def get_process_memory_gb(self) -> float:
        """Returns current process Resident Set Size (RSS) in GB."""
        try:
            mem_bytes = self.process.memory_info().rss
            return round(mem_bytes / (1024**3), 3)
        except Exception:
            return 0.0

    def sample_peak(self) -> float:
        """Samples current memory and updates peak process RAM."""
        current = self.get_process_memory_gb()
        if current > self.peak_memory_gb:
            self.peak_memory_gb = current
        return self.peak_memory_gb

    def get_resource_snapshot(self) -> Dict[str, Any]:
        """Returns complete memory snapshot for benchmark logging."""
        self.sample_peak()
        avail_mem = HardwareMonitor.get_available_memory_gb()
        swap_used = HardwareMonitor.get_swap_usage_gb()

        return {
            "process_memory_gb": self.get_process_memory_gb(),
            "peak_process_memory_gb": self.peak_memory_gb,
            "baseline_memory_gb": self.baseline_memory_gb,
            "available_system_memory_gb": avail_mem,
            "swap_used_gb": swap_used,
        }
