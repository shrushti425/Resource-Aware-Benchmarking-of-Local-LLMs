"""Base Task Interface for LLM Benchmark Tasks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class BaseTask(ABC):
    """Abstract base class for all benchmark tasks."""

    def __init__(self, task_name: str, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        self.task_name = task_name
        self.dataset_path = dataset_path
        if prompt_path is None:
            # Default to prompts/<task_name>/default.txt
            root_prompts = Path(__file__).resolve().parent.parent.parent.parent / "prompts" / task_name / "default.txt"
            if root_prompts.exists():
                prompt_path = root_prompts

        self.prompt_path = prompt_path
        self.prompt_template = self._load_prompt()
        self.dataset: List[Dict[str, Any]] = []

    def _load_prompt(self) -> str:
        """Loads prompt template from prompts directory or uses fallback default."""
        if self.prompt_path and Path(self.prompt_path).exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        # Fallback templates per task
        fallbacks = {
            "qa": "Context: {context}\nQuestion: {question}\nAnswer:",
            "classification": "Categories: {categories}\nText: {text}\nCategory:",
            "summarization": "Text: {text}\nSummary:",
            "extraction": "Extract {entity_type} from: {text}\nExtracted:",
            "json": "Schema: {schema_description}\nText: {text}\nJSON:",
            "reasoning": "Problem: {question}\nReasoning:",
            "rag": "Retrieved: {retrieved_chunks}\nQuestion: {question}\nAnswer:",
        }
        return fallbacks.get(self.task_name, "Input: {text}\nOutput:")

    @abstractmethod
    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads and returns dataset samples up to sample_limit."""
        pass

    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Formats dataset sample into model prompt string."""
        pass

    @abstractmethod
    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates single prediction against sample reference."""
        pass

    def evaluate_batch(self, predictions: List[str], samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluates batch of predictions and returns mean task metrics."""
        if not predictions or not samples or len(predictions) != len(samples):
            return {}

        metric_sums: Dict[str, float] = {}
        for pred, sample in zip(predictions, samples):
            m = self.evaluate_sample(pred, sample)
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

        n = len(predictions)
        return {k: round(v / n, 4) for k, v in metric_sums.items()}
