"""Unit tests for configuration loading."""

import pytest
from benchmark.config import BenchmarkConfig


def test_benchmark_config_loading():
    cfg = BenchmarkConfig()
    assert len(cfg.models) > 0
    assert cfg.defaults.temperature == 0.0
    assert cfg.defaults.seed == 42
    assert cfg.hardware.total_memory_gb == 8.0


def test_get_model():
    cfg = BenchmarkConfig()
    m = cfg.get_model("llama-3.2-1b-q4")
    assert m is not None
    assert m.family == "Llama"
    assert m.quantization == "Q4_K_M"
