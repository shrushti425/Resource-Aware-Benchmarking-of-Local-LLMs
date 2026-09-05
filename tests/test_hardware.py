"""Unit tests for hardware monitor."""

import pytest
from benchmark.hardware import HardwareMonitor


def test_hardware_detection():
    hw = HardwareMonitor.detect_hardware()
    assert "chip" in hw
    assert "total_unified_memory_gb" in hw
    assert hw["total_unified_memory_gb"] > 0
    assert "metal_available" in hw


def test_memory_feasibility_check():
    # 2GB model on 8GB machine -> Feasible
    ok_small, _ = HardwareMonitor.check_memory_feasibility(2.0, safety_margin_gb=1.5)
    assert ok_small is True

    # 10GB model on 8GB machine -> Infeasible
    ok_huge, reason = HardwareMonitor.check_memory_feasibility(10.0, safety_margin_gb=1.5)
    assert ok_huge is False
    assert "exceeds safe memory limit" in reason
