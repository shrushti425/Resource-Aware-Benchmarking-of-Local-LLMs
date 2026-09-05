"""Text Classification Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.metrics.text import classification_metrics, exact_match_score


DEFAULT_CLASSIFICATION_SAMPLES = [
    {
        "id": "cls_001",
        "text": "Apple released the M2 chip featuring an 8-core CPU and up to 10-core GPU on ARM architecture.",
        "categories": "Technology, Finance, Sports, Politics",
        "reference": "Technology",
    },
    {
        "id": "cls_002",
        "text": "Federal Reserve kept interest rates steady today, affecting stock indices across global financial markets.",
        "categories": "Technology, Finance, Sports, Politics",
        "reference": "Finance",
    },
    {
        "id": "cls_003",
        "text": "The national team secured a 3-1 victory in the championship final after scoring two second-half goals.",
        "categories": "Technology, Finance, Sports, Politics",
        "reference": "Sports",
    },
    {
        "id": "cls_004",
        "text": "Parliament voted to approve the new infrastructure expenditure bill following a multi-day debate.",
        "categories": "Technology, Finance, Sports, Politics",
        "reference": "Politics",
    },
]


class ClassificationTask(BaseTask):
    """Categorical text classification benchmark task."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("classification", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_CLASSIFICATION_SAMPLES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(categories=sample["categories"], text=sample["text"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        ref = sample["reference"]
        return {
            "accuracy": exact_match_score(prediction, ref),
        }

    def evaluate_batch(self, predictions: List[str], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        refs = [s["reference"] for s in samples]
        return classification_metrics(predictions, refs)
