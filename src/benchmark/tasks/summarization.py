"""Text Summarization Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.metrics.text import calculate_rouge_scores


DEFAULT_SUMMARIZATION_SAMPLES = [
    {
        "id": "sum_001",
        "text": "Running large language models locally on consumer hardware requires careful resource management. On an 8GB Apple Silicon MacBook Air, system memory is shared dynamically between CPU, GPU, and macOS operating system overhead. Quantized GGUF models running via llama.cpp allow execution without crashing system memory boundaries.",
        "reference": "Local LLM inference on an 8GB Apple Silicon MacBook Air requires quantized GGUF models and llama.cpp to efficiently share unified memory without triggering macOS memory pressure.",
    },
    {
        "id": "sum_002",
        "text": "Retrieval-Augmented Generation (RAG) enhances language models by retrieving relevant document snippets from a local vector index before generating answers. Fixed retrieval benchmarks keep the embedding model and vector store constant while varying the local LLM generation engine to measure answer quality and latency.",
        "reference": "RAG systems retrieve relevant document chunks from a vector store to ground LLM answers, allowing comparative evaluation of local generation engines under fixed retrieval conditions.",
    },
]


class SummarizationTask(BaseTask):
    """Text summarization benchmark task evaluating ROUGE-1, ROUGE-2, and ROUGE-L."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("summarization", dataset_path, prompt_path)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        if not samples:
            samples = DEFAULT_SUMMARIZATION_SAMPLES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return self.prompt_template.format(text=sample["text"])

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        return calculate_rouge_scores(prediction, sample["reference"])
