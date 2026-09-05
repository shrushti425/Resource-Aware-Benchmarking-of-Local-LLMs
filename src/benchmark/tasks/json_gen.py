"""Structured JSON Generation Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.metrics.json_eval import evaluate_json_output


DEFAULT_JSON_SAMPLES = [
    {
        "id": "json_001",
        "schema_description": "JSON object with keys 'name', 'architecture', 'unified_memory_gb', and 'status'.",
        "expected_keys": ["name", "architecture", "unified_memory_gb", "status"],
        "text": "Device name is Apple M2 built on ARM64 architecture with 8GB unified memory and status active.",
        "reference_dict": {
            "name": "Apple M2",
            "architecture": "ARM64",
            "unified_memory_gb": 8,
            "status": "active",
        },
    },
]


class StructuredGenerationTask(BaseTask):
    """Structured JSON generation benchmark task evaluating JSON validity and schema compliance."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("json", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_JSON_SAMPLES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(schema_description=sample["schema_description"], text=sample["text"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        expected_keys = sample.get("expected_keys", [])
        ref_dict = sample.get("reference_dict", None)
        return evaluate_json_output(prediction, expected_keys, ref_dict)
