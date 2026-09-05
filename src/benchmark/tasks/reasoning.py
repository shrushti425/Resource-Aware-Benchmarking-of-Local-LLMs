"""Reasoning and Math Logic Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
from benchmark.tasks.base import BaseTask
from benchmark.metrics.text import exact_match_score


DEFAULT_REASONING_SAMPLES = [
    {
        "id": "reas_001",
        "question": "A model file size is 4 GB. If the laptop has 8 GB total memory and macOS uses 3 GB, how much free memory remains after loading the model?",
        "reference": "1 GB",
    },
    {
        "id": "reas_002",
        "question": "If an LLM generates 120 output tokens in 4 seconds, what is its throughput in tokens per second?",
        "reference": "30",
    },
]


class ReasoningTask(BaseTask):
    """Reasoning and multi-step logic benchmark task."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("reasoning", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_REASONING_SAMPLES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(question=sample["question"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        ref = sample["reference"]

        # Parse Final Answer: if present
        pred_answer = prediction
        match = re.search(r"Final Answer:\s*(.*)", prediction, re.IGNORECASE)
        if match:
            pred_answer = match.group(1).strip()

        em = exact_match_score(pred_answer, ref)
        return {
            "exact_match": em,
            "accuracy": em,
        }
