"""Apple Silicon Hardware and Memory Monitoring Module."""

import os
import platform
import subprocess
from typing import Any, Dict, Tuple
import psutil


class HardwareMonitor:
    """Hardware detector and resource tracking engine for Apple Silicon."""

    @staticmethod
    def get_chip_info() -> str:
        """Returns the Apple Silicon chip brand string (e.g. Apple M2)."""
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            if val:
                return val
        except Exception:
            pass
        return platform.processor() or "Apple Silicon"

    @staticmethod
    def get_macos_version() -> str:
        """Returns the exact macOS version."""
        try:
            res = subprocess.run(
                ["sysctl", "-n", "kern.osproductversion"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            if val:
                return val
        except Exception:
            pass
        return platform.mac_ver()[0] or "macOS"

    @staticmethod
    def get_unified_memory_gb() -> float:
        """Returns total physical unified memory in GB."""
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            if val and val.isdigit():
                return round(int(val) / (1024**3), 2)
        except Exception:
            pass
        return round(psutil.virtual_memory().total / (1024**3), 2)

    @staticmethod
    def get_available_memory_gb() -> float:
        """Returns available system memory in GB."""
        vm = psutil.virtual_memory()
        return round(vm.available / (1024**3), 2)

    @staticmethod
    def get_swap_usage_gb() -> float:
        """Returns current swap usage in GB."""
        try:
            swap = psutil.swap_memory()
            return round(swap.used / (1024**3), 2)
        except Exception:
            return 0.0

    @staticmethod
    def is_metal_available() -> bool:
        """Returns True if running on macOS Apple Silicon with Metal support."""
        if platform.system() != "Darwin":
            return False
        # Check arm64 or Apple Silicon processor
        arch = platform.machine()
        return arch == "arm64" or "Apple" in HardwareMonitor.get_chip_info()

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Returns complete hardware diagnostic snapshot."""
        chip = HardwareMonitor.get_chip_info()
        macos_ver = HardwareMonitor.get_macos_version()
        total_ram = HardwareMonitor.get_unified_memory_gb()
        avail_ram = HardwareMonitor.get_available_memory_gb()
        swap_used = HardwareMonitor.get_swap_usage_gb()
        metal_ok = HardwareMonitor.is_metal_available()

        return {
            "chip": chip,
            "architecture": platform.machine(),
            "os": f"macOS {macos_ver}",
            "total_unified_memory_gb": total_ram,
            "available_memory_gb": avail_ram,
            "swap_used_gb": swap_used,
            "metal_available": metal_ok,
            "cpu_count_logical": os.cpu_count() or 8,
        }

    @staticmethod
    def check_memory_feasibility(model_file_size_gb: float, safety_margin_gb: float = 1.5) -> Tuple[bool, str]:
        """Determines whether a model can fit safely in memory without memory pressure failure."""
        total_ram = HardwareMonitor.get_unified_memory_gb()
        avail_ram = HardwareMonitor.get_available_memory_gb()

        # On an 8GB machine, a model larger than 4.5GB will cause severe memory pressure
        if model_file_size_gb > (total_ram - safety_margin_gb):
            return (
                False,
                f"Model file size ({model_file_size_gb:.2f} GB) exceeds safe memory limit for {total_ram:.1f} GB unified memory (safety margin {safety_margin_gb:.1f} GB).",
            )

        if avail_ram < model_file_size_gb:
            return (
                False,
                f"Available system memory ({avail_ram:.2f} GB) is lower than required model size ({model_file_size_gb:.2f} GB).",
            )

        return True, "Feasible"
