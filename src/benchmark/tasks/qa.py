"""Question Answering Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.metrics.text import exact_match_score, f1_score


DEFAULT_QA_SAMPLES = [
    {
        "id": "qa_001",
        "context": "Apple Silicon M2 uses a unified memory architecture, allowing the CPU, GPU, and Neural Engine to access high-bandwidth memory directly without PCIe transfer overhead.",
        "question": "What architecture does Apple Silicon M2 use for shared RAM access?",
        "reference": "unified memory architecture",
    },
    {
        "id": "qa_002",
        "context": "GGUF is a binary file format designed for fast model loading and quantization compatibility with llama.cpp on Apple Silicon Metal acceleration.",
        "question": "Which inference runtime uses GGUF model format on macOS?",
        "reference": "llama.cpp",
    },
    {
        "id": "qa_003",
        "context": "Quantization reduces precision from 16-bit float to 4-bit integer (Q4_K_M), reducing model footprint from 14GB down to 4GB.",
        "question": "What is the memory reduction factor of 4-bit quantization?",
        "reference": "from 14GB down to 4GB",
    },
    {
        "id": "qa_004",
        "context": "Unified memory on an 8GB MacBook Air limits safe model allocation to approximately 4GB to prevent macOS memory pressure and disk swap thrashing.",
        "question": "What is the safe model memory limit on an 8GB MacBook Air?",
        "reference": "approximately 4GB",
    },
    {
        "id": "qa_005",
        "context": "Metal API is Apple's hardware-accelerated graphics and compute framework utilized by llama.cpp for GPU acceleration on M1 and M2 chips.",
        "question": "Which framework provides GPU acceleration on Apple Silicon?",
        "reference": "Metal API",
    },
]


class QATask(BaseTask):
    """Question Answering benchmark task evaluating Exact Match and F1 score."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("qa", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_QA_SAMPLES

        # Extend synthetically if requested count exceeds defaults
        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            extended = (samples * mult)[:sample_limit]
            for i, s in enumerate(extended):
                s_copy = dict(s)
                s_copy["id"] = f"{s['id']}_sample_{i}"
                samples.append(s_copy)

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(context=sample["context"], question=sample["question"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        ref = sample["reference"]
        return {
            "exact_match": exact_match_score(prediction, ref),
            "f1": f1_score(prediction, ref),
        }
