"""Information Extraction Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.metrics.text import f1_score


DEFAULT_EXTRACTION_SAMPLES = [
    {
        "id": "ext_001",
        "entity_type": "Hardware Technologies",
        "text": "The MacBook Air features the Apple M2 chip, integrated Metal GPU, and 8GB Unified Memory.",
        "reference": "Apple M2, Metal GPU, Unified Memory",
    },
    {
        "id": "ext_002",
        "entity_type": "Software Libraries",
        "text": "We evaluated llama-cpp-python, FAISS, Streamlit, PyTest, and Matplotlib for our benchmarking framework.",
        "reference": "llama-cpp-python, FAISS, Streamlit, PyTest, Matplotlib",
    },
]


class InformationExtractionTask(BaseTask):
    """Information and named entity extraction benchmark task."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("extraction", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_EXTRACTION_SAMPLES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(entity_type=sample["entity_type"], text=sample["text"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        return {
            "f1": f1_score(prediction, sample["reference"]),
        }
