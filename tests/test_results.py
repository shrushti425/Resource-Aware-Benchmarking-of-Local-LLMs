"""Unit tests for JSONL result storage and resume manager."""

import pytest
from benchmark.results.storage import ResultStorage
from benchmark.results.resume import ResumeManager


def test_result_storage_roundtrip(tmp_path):
    storage = ResultStorage(results_dir=tmp_path)
    rec = {
        "run_id": "test_run",
        "model": "test-model",
        "task": "qa",
        "sample_id": "s1",
        "latency": 0.5,
    }
    storage.append_result(rec, filename="test.jsonl")

    loaded = storage.load_all_results(filename="test.jsonl")
    assert len(loaded) == 1
    assert loaded[0]["model"] == "test-model"

    resume_mgr = ResumeManager(storage, jsonl_filename="test.jsonl")
    assert resume_mgr.is_completed("test-model", "qa", "s1") is True
    assert resume_mgr.is_completed("test-model", "qa", "s2") is False
