"""Retrieval-Augmented Generation (RAG) Benchmark Task."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from benchmark.tasks.base import BaseTask
from benchmark.rag.pipeline import RAGPipeline
from benchmark.metrics.rag_eval import calculate_retrieval_metrics, evaluate_rag_generation


DEFAULT_RAG_DOCUMENTS = [
    {
        "doc_id": "doc_01",
        "title": "Unified Memory Architecture",
        "text": "Apple Silicon M2 unified memory architecture enables the CPU, GPU, and Neural Engine to address a single pool of high-bandwidth memory without copying buffers across PCIe lanes.",
    },
    {
        "doc_id": "doc_02",
        "title": "llama.cpp Metal Backend",
        "text": "llama.cpp accelerates GGUF model execution on macOS via Apple Metal performance shaders, drastically reducing latency for 1B-4B parameter models.",
    },
    {
        "doc_id": "doc_03",
        "title": "8GB Memory Safety Boundaries",
        "text": "When running local LLMs on an 8GB RAM MacBook Air, keeping process memory under 4.0 GB prevents disk swapping and macOS kernel memory pressure.",
    },
]

DEFAULT_RAG_QUERIES = [
    {
        "id": "rag_001",
        "question": "What is the primary benefit of unified memory on Apple Silicon M2?",
        "ground_truth_doc_ids": ["doc_01"],
        "reference_answer": "CPU, GPU, and Neural Engine share a single high-bandwidth memory pool without PCIe copying overhead.",
    },
    {
        "id": "rag_002",
        "question": "How does llama.cpp achieve GPU acceleration on macOS?",
        "ground_truth_doc_ids": ["doc_02"],
        "reference_answer": "Via Apple Metal performance shaders accelerating GGUF model execution.",
    },
]


class RAGTask(BaseTask):
    """RAG benchmark task evaluating retrieval accuracy (Recall@K, MRR) and generation answer quality."""

    def __init__(self, dataset_path: Optional[Path] = None, prompt_path: Optional[Path] = None):
        super().__init__("rag", dataset_path, prompt_path)
        self.rag_pipeline = RAGPipeline()
        self.documents = DEFAULT_RAG_DOCUMENTS
        self.rag_pipeline.index_documents(self.documents)

    def load_dataset(self, sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "documents" in data:
                        self.documents = data["documents"]
                        self.rag_pipeline = RAGPipeline()
                        self.rag_pipeline.index_documents(self.documents)
                    samples = data.get("samples", [])
                elif isinstance(data, list):
                    samples = data

        if not samples:
            samples = DEFAULT_RAG_QUERIES

        if sample_limit and len(samples) < sample_limit:
            mult = (sample_limit // len(samples)) + 1
            samples = (samples * mult)[:sample_limit]

        self.dataset = samples[:sample_limit] if sample_limit else samples
        return self.dataset

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        # Prompt is formatted dynamically by RAGPipeline after chunk retrieval
        return sample["question"]

    def evaluate_sample(self, prediction: str, sample: Dict[str, Any]) -> Dict[str, float]:
        gt_doc_ids = sample.get("ground_truth_doc_ids", [])
        ref_ans = sample.get("reference_answer", "")

        # Retrieval metrics on retrieved doc IDs attached in metadata if present
        retrieved_ids = sample.get("retrieved_doc_ids", [])
        ret_metrics = calculate_retrieval_metrics(retrieved_ids, gt_doc_ids, ks=[1, 3, 5])
        gen_metrics = evaluate_rag_generation(prediction, ref_ans)

        return ret_metrics | gen_metrics
