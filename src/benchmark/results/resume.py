"""Checkpoint and Resume Manager for long-running benchmarks."""

from typing import Set, Tuple
from benchmark.results.storage import ResultStorage


class ResumeManager:
    """Tracks completed experiment combinations to skip already computed tasks."""

    def __init__(self, storage: ResultStorage, jsonl_filename: str = "benchmark_results.jsonl"):
        self.storage = storage
        self.jsonl_filename = jsonl_filename
        self.completed_keys: Set[Tuple[str, str, str]] = set()
        self._scan_completed()

    def _scan_completed(self):
        """Scans existing JSONL file for completed (model, task, sample_id) tuples."""
        records = self.storage.load_all_results(self.jsonl_filename)
        for r in records:
            model = r.get("model")
            task = r.get("task")
            sample_id = r.get("sample_id")
            if model and task and sample_id:
                self.completed_keys.add((str(model), str(task), str(sample_id)))

    def is_completed(self, model_id: str, task_name: str, sample_id: str) -> bool:
        """Returns True if this exact sample has already been evaluated."""
        return (str(model_id), str(task_name), str(sample_id)) in self.completed_keys
