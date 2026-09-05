"""Unit tests for model registry."""

import pytest
from benchmark.models.registry import ModelRegistry


def test_model_registry_list():
    reg = ModelRegistry()
    enabled = reg.list_models(include_disabled=False)
    all_models = reg.list_models(include_disabled=True)
    assert len(all_models) >= len(enabled)
    assert any(m.id == "llama-3.2-1b-q4" for m in enabled)


def test_model_registry_verify():
    reg = ModelRegistry()
    ver = reg.verify_models()
    assert "llama-3.2-1b-q4" in ver
    assert ver["llama-3.2-1b-q4"]["quantization"] == "Q4_K_M"
